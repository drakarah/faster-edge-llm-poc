"""Layer-by-layer GPTQ / GPTAQ (INT4, group 128, symmetric) of the DENSE linears of Qwen3.6-35B-A3B, on top of the already quantised experts.

Which linears (--parts, comma separated): shared (shared_expert gate/up/down of every layer), head (lm_head), gdn (in_proj_qkv, in_proj_z, out_proj of the 30 gated-delta-net layers),
attn (q/k/v/o of the 10 full-attention layers). in_proj_a / in_proj_b (32 rows), the router, norms and embeddings are never quantised. Attention parts can be restricted to some layers
(--attn-layers 3,7,... = quantise only these full-attention layer indices; default all).
Streams per layer (the experts are NOT re-quantised, they come from the overlay written by gptq_experts_layerwise_qwen36.py, EXPERTS dir):
  X  : activations produced by the already quantised layers (dense parts selected here + overlay experts, dequantised) = what the final model sees;
  X~ : activations of the full-precision model (BF16 source layers) on the same calibration tokens (only with --gptaq).
Pass 1 (original dense weights, layer in BF16): the layer runs on X and (GPTAQ) on X~; hooks on every selected linear collect H = X_in^T X_in and D = (X~_in - X_in)^T X_in; the X~ output is the next X~.
The linears are quantised (tools/gptaq.py: GPTQ when --gptaq is off, GPTAQ otherwise, MSE-clipped scales), pass 2 runs the layer with the quantised dense weights and the dequantised overlay experts on X to give the next X.
The per-linear relative weight error on the X-stream inputs, ||(W_q - W) X||^2 / ||W X||^2 (the quantity GPTQ minimises; GPTAQ trades some of it for the X~ target), goes to conversion_report_dense.json. --holdout N keeps the last N sequences out of every Hessian.
Output DST = hard links of EXPERTS (source shards + expert shards) + model-dense-int4.safetensors + index/config updated (quantised linears lose their `.weight`).
Usage (PoC image, GPU): gptq_dense_layerwise_qwen36.py SRC EXPERTS DST --text-dir /text --parts shared,head,gdn,attn [--gptaq] [--samples 256] [--seq-len 1024] [--layers N] [--holdout 32]"""
import argparse, json, os, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

for p in ('/tools', os.path.dirname(os.path.abspath(__file__)),
          os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'faster-llm-poc', 'poc', 'tools')):
    if os.path.isdir(p):
        sys.path.insert(0, p)
from gptq_dense_layerwise_26b import calibration_ids  # noqa: E402
from gptaq import hinv_upper, gptaq_rows, p_matrix, G  # noqa: E402
from ct_to_gptq import pack_gptq  # noqa: E402


def unpack_gptq(qweight, scales):
    """GPTQ (zero point 8) tensors [K/8, N] int32 + [K/G, N] fp16 -> dequantised weight [N, K] bf16."""
    k8, n = qweight.shape
    q = ((qweight.unsqueeze(1) >> (4 * torch.arange(8, device=qweight.device).view(1, 8, 1))) & 0xF).reshape(k8 * 8, n).float() - 8.0
    return (q * scales.float().repeat_interleave(G, 0)).t().contiguous().to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('experts'); ap.add_argument('dst'); ap.add_argument('--text-dir', required=True)
    ap.add_argument('--parts', default='shared,head'); ap.add_argument('--attn-layers', default=''); ap.add_argument('--gptaq', action='store_true')
    ap.add_argument('--samples', type=int, default=256); ap.add_argument('--seq-len', type=int, default=1024); ap.add_argument('--damp', type=float, default=0.01)
    ap.add_argument('--layers', type=int, default=0); ap.add_argument('--mb', type=int, default=4); ap.add_argument('--holdout', type=int, default=0)
    a = ap.parse_args(); dev = 'cuda'; parts = set(a.parts.split(','))
    attn_only = {int(x) for x in a.attn_layers.split(',') if x}
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    cfg = AutoConfig.from_pretrained(a.src).text_config; cfg._attn_implementation = 'sdpa'
    tok = AutoTokenizer.from_pretrained(a.src); ids = calibration_ids(a.text_dir, tok, a.samples, a.seq_len)
    wm = json.load(open(f'{a.src}/model.safetensors.index.json'))['weight_map']; em = json.load(open(f'{a.experts}/model.safetensors.index.json'))['weight_map']; handles = {}
    def get(name, m=wm, root=a.src):
        sh = m[name]
        if (root, sh) not in handles: handles[(root, sh)] = safe_open(f'{root}/{sh}', framework='pt')
        return handles[(root, sh)].get_tensor(name)
    P_ = 'model.language_model.'; N, T = ids.shape; H_, I = cfg.hidden_size, cfg.moe_intermediate_size; n_cal = (N - a.holdout) * T
    rotary = M.Qwen3_5MoeTextRotaryEmbedding(cfg).to(dev); pos4 = torch.arange(T, device=dev).view(1, 1, -1).expand(4, a.mb, -1); text_pos = pos4[0]; rope_pos = pos4[1:]
    emb = torch.nn.Embedding(cfg.vocab_size, H_).to(dev, torch.bfloat16); emb.weight.data.copy_(get(P_ + 'embed_tokens.weight').to(dev))
    with torch.no_grad(): hidden = torch.cat([emb(ids[i:i + a.mb].to(dev)) for i in range(0, N, a.mb)])
    del emb; use_fp = a.gptaq or a.holdout > 0; hidden_fp = hidden.cpu() if use_fp else None   # the full-precision stream lives in host memory (1 GB); micro-batches go through the GPU; torch.cuda.empty_cache()
    n_layers = a.layers or cfg.num_hidden_layers; out, report = {}, {}; t0 = time.time()

    def selected(layer, l):
        names = []
        if 'shared' in parts: names += ['mlp.shared_expert.gate_proj', 'mlp.shared_expert.up_proj', 'mlp.shared_expert.down_proj']
        if 'gdn' in parts and layer.block_type == 'linear_attention': names += ['linear_attn.in_proj_qkv', 'linear_attn.in_proj_z', 'linear_attn.out_proj']
        if 'attn' in parts and layer.block_type == 'full_attention' and (not attn_only or l in attn_only): names += [f'self_attn.{x}_proj' for x in 'qkvo']
        return names

    def run_layer(layer, hs, hooks_on=None):
        b = hs.shape[0]; cos_sin = rotary(hs, rope_pos[:, :b])
        h = layer.input_layernorm(hs)
        if layer.block_type == 'linear_attention': h = layer.linear_attn(hidden_states=h, cache_params=None, attention_mask=None)
        else: h, _ = layer.self_attn(hidden_states=h, attention_mask=None, position_ids=text_pos[:b], past_key_values=None, position_embeddings=cos_sin)
        mid = hs + h; return mid + layer.mlp(layer.post_attention_layernorm(mid))

    for l in range(n_layers):
        layer = M.Qwen3_5MoeDecoderLayer(cfg, l).to(dev, torch.bfloat16).eval(); pre = f'{P_}layers.{l}.'
        layer.load_state_dict({k[len(pre):]: get(k) for k in wm if k.startswith(pre)}, strict=False)
        names = selected(layer, l); mods = {n: layer.get_submodule(n) for n in names}
        Hs = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in mods.items()}
        Ds = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in mods.items()} if a.gptaq else {}
        cur = {'seq0': 0}
        # the two streams run one after the other on the same micro-batch so x and x~ of one tokens block are both at hand
        saved_x = {}
        def hook_x(n):
            def f(mod, inp):
                x = inp[0].reshape(-1, inp[0].shape[-1]); keep = max(0, min(x.shape[0], n_cal - cur['seq0'] * T))
                if keep: xf = x[:keep].float(); Hs[n] += xf.t() @ xf; saved_x[n] = xf
            return f
        def hook_xt(n):
            def f(mod, inp):
                x = inp[0].reshape(-1, inp[0].shape[-1]); keep = max(0, min(x.shape[0], n_cal - cur['seq0'] * T))
                if keep: Ds[n] += (x[:keep].float() - saved_x[n]).t() @ saved_x[n]
            return f
        with torch.no_grad():
            next_fp = torch.empty_like(hidden_fp) if use_fp else None
            for i in range(0, N, a.mb):
                cur['seq0'] = i; hs = hidden[i:i + a.mb]
                hx = [m.register_forward_pre_hook(hook_x(n)) for n, m in mods.items()]; run_layer(layer, hs)
                for h in hx: h.remove()
                if a.gptaq:
                    ht = [m.register_forward_pre_hook(hook_xt(n)) for n, m in mods.items()]; next_fp[i:i + a.mb] = run_layer(layer, hidden_fp[i:i + a.mb].to(dev)).cpu()
                    for h in ht: h.remove()
                elif use_fp: next_fp[i:i + a.mb] = run_layer(layer, hidden_fp[i:i + a.mb].to(dev)).cpu()
                saved_x.clear()
        errs = {}
        for n, m in mods.items():
            w = m.weight.data.float(); U = hinv_upper(Hs[n], a.damp); Pm = p_matrix(Ds[n], U) if a.gptaq else None
            q, sc = gptaq_rows(w, U, Pm); deq = q.float() * sc.repeat_interleave(G, 1)
            tgt_in = Hs[n]; e = deq - w  # error on the calibration inputs (X stream); with GPTAQ the target moves to W X~, reported on the X-stream proxy
            errs[f'{pre}{n}'] = (((e @ tgt_in) * e).sum() / ((w @ tgt_in) * w).sum().clamp_min(1e-12)).item()
            for k, t in pack_gptq(q.cpu(), sc.cpu()).items(): out[f'{pre}{n}.{k}'] = t
            m.weight.data.copy_(deq.to(m.weight.dtype))
        del Hs, Ds
        # quantised stream: dequantised overlay experts into the layer
        ex = layer.mlp.experts
        with torch.no_grad():
            for e_ in range(cfg.num_experts):
                base = f'{pre}mlp.experts.{e_}.'
                g = unpack_gptq(get(base + 'gate_proj.qweight', em, a.experts).to(dev), get(base + 'gate_proj.scales', em, a.experts).to(dev))
                u = unpack_gptq(get(base + 'up_proj.qweight', em, a.experts).to(dev), get(base + 'up_proj.scales', em, a.experts).to(dev))
                d = unpack_gptq(get(base + 'down_proj.qweight', em, a.experts).to(dev), get(base + 'down_proj.scales', em, a.experts).to(dev))
                ex.gate_up_proj.data[e_].copy_(torch.cat([g, u], 0)); ex.down_proj.data[e_].copy_(d)
            nxt = torch.empty_like(hidden)
            for i in range(0, N, a.mb): nxt[i:i + a.mb] = run_layer(layer, hidden[i:i + a.mb])
        hidden = nxt
        dev_tr = dev_ho = None
        if use_fp:   # deviation of the quantised stream from the full-precision stream after this layer: ||X - X~|| / ||X~||, calibration and held-out sequences separately
            def rel(a_, b_):   # in chunks of sequences: a float copy of all calibration activations does not fit next to the layer
                num = den = 0.0
                for j in range(0, a_.shape[0], 8): bb = b_[j:j + 8].to(dev).float(); num += (a_[j:j + 8].float() - bb).pow(2).sum().item(); den += bb.pow(2).sum().item()
                return (num / max(den, 1e-12)) ** 0.5
            hidden_fp = next_fp; dev_tr = rel(hidden[:N - a.holdout], hidden_fp[:N - a.holdout]); dev_ho = rel(hidden[N - a.holdout:], hidden_fp[N - a.holdout:]) if a.holdout else None
        report[f'layer{l}'] = {'linear_relative_error': errs, 'stream_deviation_calibration': dev_tr, 'stream_deviation_heldout': dev_ho}
        mean_e = sum(errs.values()) / max(1, len(errs))
        print(f'layer {l} ({layer.block_type}): {len(errs)} linears quantised, mean weight error {mean_e:.4f}, stream deviation calib {dev_tr} held-out {dev_ho}  {time.time() - t0:.0f} s', flush=True)
        del layer; torch.cuda.empty_cache()
    if 'head' in parts and n_layers == cfg.num_hidden_layers:
        norm = M.Qwen3_5MoeRMSNorm(H_, eps=cfg.rms_norm_eps).to(dev, torch.bfloat16); norm.weight.data.copy_(get(P_ + 'norm.weight').to(dev))
        import gc; gc.collect(); torch.cuda.empty_cache()   # last layer's Hessians/hooks must not sit next to the 1.9 GB lm_head (D ran out of GPU memory here)
        lm = get('lm_head.weight').to(dev); Hh = torch.zeros(H_, H_, device=dev); Dh = torch.zeros(H_, H_, device=dev)
        with torch.no_grad():
            for i in range(0, N, a.mb):
                x = norm(hidden[i:i + a.mb]).reshape(-1, H_).float()[:max(0, n_cal - i * T)]
                if x.shape[0]: Hh += x.t() @ x
                if a.gptaq and x.shape[0]: xt = norm(hidden_fp[i:i + a.mb].to(dev)).reshape(-1, H_).float()[:x.shape[0]]; Dh += (xt - x).t() @ x
        U = hinv_upper(Hh, a.damp); Pm = p_matrix(Dh, U) if a.gptaq else None; qs, ss = [], []
        for s in range(0, lm.shape[0], 8192):
            q, sc = gptaq_rows(lm[s:s + 8192].float(), U, Pm); qs.append(q.cpu()); ss.append(sc.cpu())
        for k, t in pack_gptq(torch.cat(qs), torch.cat(ss)).items(): out[f'lm_head.{k}'] = t
        print('lm_head quantised', flush=True)
    os.makedirs(a.dst, exist_ok=True)
    for f in os.listdir(a.experts):
        d = f'{a.dst}/{f}'
        if os.path.exists(d): os.remove(d)
        if f.endswith('.safetensors'): os.link(f'{a.experts}/{f}', d)
        elif f not in ('model.safetensors.index.json', 'config.json', 'conversion_report.json'): shutil.copy(f'{a.experts}/{f}', d)
    save_file(out, f'{a.dst}/model-dense-int4.safetensors', metadata={'format': 'pt'})
    ix = json.load(open(f'{a.experts}/model.safetensors.index.json')); wm2 = ix['weight_map']
    for k in [k for k in out if k.endswith('.qweight')]: wm2.pop(k.rsplit('.', 1)[0] + '.weight', None)
    for k in out: wm2[k] = 'model-dense-int4.safetensors'
    json.dump(ix, open(f'{a.dst}/model.safetensors.index.json', 'w')); shutil.copy(f'{a.experts}/config.json', f'{a.dst}/config.json')
    json.dump(report, open(f'{a.dst}/conversion_report_dense.json', 'w'))
    print('GPTQ_DENSE_DONE', len(out) // 3, 'linears', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()

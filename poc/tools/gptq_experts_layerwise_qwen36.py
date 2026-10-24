"""Layer-by-layer calibrated GPTQ (INT4, group 128, symmetric, MSE-clipped scales) of the ROUTED EXPERTS of Qwen3.6-35B-A3B (256 experts x 40 layers, fused BF16 tensors in the checkpoint).

Why a custom loop: the BF16 checkpoint is 72 GB (does not fit RAM or VRAM). One decoder layer at a time is built from the shards on the GPU. Per layer:
  1. the token mixer (gated delta net or gated attention) and the post-attention norm run on the calibration activations (N x seq x hidden); the MoE input, the residual stream and the router's top-8 choice per token are kept;
  2. the experts are processed in groups: per expert E the Hessian H = X^T X of the tokens ROUTED to it is accumulated for gate/up (shared input), gate and up are quantised with GPTQ (tools/gptq_dense_layerwise_26b.py `gptq_linear`), then the expert's down_proj input (silu(gate) * up computed WITH the quantised gate/up) gives the second Hessian and down is quantised;
  3. experts that saw fewer than --min-tokens calibration tokens fall back to round-to-nearest with MSE-clipped scales (an identity Hessian), counted in the report;
  4. the layer is run again with the dequantised experts to produce the next layer's input (sequential GPTQ), the per-layer relative output error of the MoE block is recorded, and the layer's quantised experts are written to their own shard (model-experts-LL.safetensors).
Dense weights (mixer projections, router, shared expert, norms, embeddings, lm_head) stay as they are in the checkpoint (BF16, the exporter converts to FP16): only the experts are lossy.
Output = hard-linked source shards + the expert shards + an index without the fused expert tensors (per-expert GPTQ keys `...mlp.experts.E.{gate,up,down}_proj.{qweight,qzeros,scales}`, the format the fork's Qwen3 sparse MoE block repacks for the INT4 MoE plugin) + config with quantization_config (gptq, g128, sym).
Calibration text: generic (nl/en/fr Wikipedia + source code), never the application prompts.
Usage (PoC image, GPU; /tools = PoC tools dir): gptq_experts_layerwise_qwen36.py SRC DST --text-dir /text [--samples 256] [--seq-len 1024] [--layers N] [--min-tokens 128]"""
import argparse, json, os, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

for p in ('/tools', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'faster-llm-poc', 'poc', 'tools')):
    if os.path.isdir(p):
        sys.path.insert(0, p)
from gptq_dense_layerwise_26b import calibration_ids, gptq_linear, G  # noqa: E402
from ct_to_gptq import pack_gptq  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--text-dir', required=True)
    ap.add_argument('--samples', type=int, default=256); ap.add_argument('--seq-len', type=int, default=1024); ap.add_argument('--damp', type=float, default=0.01)
    ap.add_argument('--layers', type=int, default=0, help='only the first N layers (test)'); ap.add_argument('--mb', type=int, default=4)
    ap.add_argument('--min-tokens', type=int, default=128); ap.add_argument('--group', type=int, default=64, help='experts per Hessian group')
    ap.add_argument('--chunk', type=int, default=32768, help='tokens per MoE chunk')
    ap.add_argument('--affinity', action='store_true', help='MoEQuant affinity-guided Hessian: each routed token counts with its gate weight (H = sum c_i x_i x_i^T)')
    ap.add_argument('--holdout', type=int, default=0, help='last N calibration sequences never enter a Hessian; the per-layer MoE error is reported on them separately')
    a = ap.parse_args(); dev = 'cuda'
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    full = AutoConfig.from_pretrained(a.src); cfg = full.text_config; cfg._attn_implementation = 'sdpa'
    tok = AutoTokenizer.from_pretrained(a.src)
    ids = calibration_ids(a.text_dir, tok, a.samples, a.seq_len)
    ix = json.load(open(f'{a.src}/model.safetensors.index.json')); wm = ix['weight_map']; handles = {}
    def get(name):
        sh = wm[name]
        if sh not in handles: handles[sh] = safe_open(f'{a.src}/{sh}', framework='pt')
        return handles[sh].get_tensor(name)
    P = 'model.language_model.'
    os.makedirs(a.dst, exist_ok=True)
    emb = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size).to(dev, torch.bfloat16); emb.weight.data.copy_(get(P + 'embed_tokens.weight').to(dev))
    rotary = M.Qwen3_5MoeTextRotaryEmbedding(cfg).to(dev)
    n_layers = a.layers or cfg.num_hidden_layers
    N, T = ids.shape; E, I, H = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
    with torch.no_grad():
        hidden = torch.cat([emb(ids[i:i + a.mb].to(dev)) for i in range(0, N, a.mb)])  # [N,T,H] bf16
    del emb; torch.cuda.empty_cache()
    pos4 = torch.arange(T, device=dev).view(1, 1, -1).expand(4, a.mb, -1); text_pos = pos4[0]; rope_pos = pos4[1:]
    report = {}; written = []; t0 = time.time()
    for l in range(n_layers):
        layer = M.Qwen3_5MoeDecoderLayer(cfg, l).to(dev, torch.bfloat16).eval()
        pre = f'{P}layers.{l}.'
        sd = {k[len(pre):]: get(k) for k in wm if k.startswith(pre)}
        miss = layer.load_state_dict(sd, strict=False)
        if l == 0: print('missing:', miss.missing_keys[:6], 'unexpected:', miss.unexpected_keys[:6], flush=True)
        del sd
        mid_cpu = torch.empty(N * T, H, dtype=torch.bfloat16); x_cpu = torch.empty(N * T, H, dtype=torch.bfloat16)
        topk_idx = torch.empty(N * T, cfg.num_experts_per_tok, dtype=torch.int16); topk_w = torch.empty(N * T, cfg.num_experts_per_tok, dtype=torch.bfloat16)
        with torch.no_grad():
            for i in range(0, N, a.mb):
                hs = hidden[i:i + a.mb]; b = hs.shape[0]
                cos_sin = rotary(hs, rope_pos[:, :b])
                h = layer.input_layernorm(hs)
                if layer.block_type == 'linear_attention':
                    h = layer.linear_attn(hidden_states=h, cache_params=None, attention_mask=None)
                else:
                    h, _ = layer.self_attn(hidden_states=h, attention_mask=None, position_ids=text_pos[:b], past_key_values=None, position_embeddings=cos_sin)
                mid = hs + h; x = layer.post_attention_layernorm(mid)
                _, w, idx = layer.mlp.gate(x.reshape(-1, H))
                s = i * T; e = s + b * T
                mid_cpu[s:e] = mid.reshape(-1, H).cpu(); x_cpu[s:e] = x.reshape(-1, H).cpu(); topk_idx[s:e] = idx.to(torch.int16).cpu(); topk_w[s:e] = w.cpu()
        experts = layer.mlp.experts
        n_cal = (N - a.holdout) * T   # tokens [0, n_cal) feed the Hessians; the rest is held out
        def routed(ti, tw, e, s0):
            m = ti == e
            sel = m.any(1).nonzero(as_tuple=True)[0]
            sel = sel[(sel + s0) < n_cal]
            wgt = (tw * m)[sel].sum(1) if a.affinity else torch.ones(sel.numel(), device=ti.device)
            return sel, wgt
        gu = experts.gate_up_proj.data; dn = experts.down_proj.data   # [E,2I,H], [E,H,I] bf16 on GPU
        tokens_per_expert = torch.bincount(topk_idx[:n_cal].reshape(-1).long(), minlength=E)
        # reference MoE output (before quantisation) for the error report
        def moe_out(chunk_slice):
            xs = x_cpu[chunk_slice].to(dev); ws = topk_w[chunk_slice].to(dev); ix_ = topk_idx[chunk_slice].long().to(dev)
            with torch.no_grad():
                y = experts(xs, ix_, ws)
            return y
        ref_sq = err_sq = ref_ho = err_ho = 0.0
        refs = []
        for s in range(0, N * T, a.chunk):
            refs.append(moe_out(slice(s, s + a.chunk)).float().cpu())
        out = {}; rtn_experts = 0; eye_h = torch.eye(H, device=dev); eye_i = torch.eye(I, device=dev)
        qgu = torch.empty_like(gu); qdn = torch.empty_like(dn)   # dequantised copies for propagation
        for g0 in range(0, E, a.group):
            grp = list(range(g0, min(E, g0 + a.group)))
            Hgu = {e: torch.zeros(H, H, device=dev) for e in grp}
            for s in range(0, N * T, a.chunk):
                xs = x_cpu[s:s + a.chunk].to(dev); ti = topk_idx[s:s + a.chunk].long().to(dev); tw = topk_w[s:s + a.chunk].float().to(dev)
                for e in grp:
                    sel, wgt = routed(ti, tw, e, s)
                    if sel.numel(): xe = xs[sel].float() * wgt.sqrt().unsqueeze(1); Hgu[e] += xe.t() @ xe
            for e in grp:
                use_eye = tokens_per_expert[e].item() < a.min_tokens; rtn_experts += int(use_eye)
                Hm = eye_h if use_eye else Hgu[e]
                for name, w in (('gate_proj', gu[e, :I]), ('up_proj', gu[e, I:])):
                    q, sc, deq, rel = gptq_linear(w.float(), Hm, a.damp)
                    for k, t in pack_gptq(q.cpu(), sc.cpu()).items(): out[f'{pre}mlp.experts.{e}.{name}.{k}'] = t
                    (qgu[e, :I] if name == 'gate_proj' else qgu[e, I:]).copy_(deq.to(torch.bfloat16))
                    report[f'{pre}mlp.experts.{e}.{name}'] = rel
            del Hgu
            Hdn = {e: torch.zeros(I, I, device=dev) for e in grp}
            for s in range(0, N * T, a.chunk):
                xs = x_cpu[s:s + a.chunk].to(dev); ti = topk_idx[s:s + a.chunk].long().to(dev); tw = topk_w[s:s + a.chunk].float().to(dev)
                for e in grp:
                    sel, wgt = routed(ti, tw, e, s)
                    if sel.numel():
                        g_, u_ = torch.nn.functional.linear(xs[sel], qgu[e]).chunk(2, dim=-1)
                        a_ = (torch.nn.functional.silu(g_) * u_).float() * wgt.sqrt().unsqueeze(1); Hdn[e] += a_.t() @ a_
            for e in grp:
                use_eye = tokens_per_expert[e].item() < a.min_tokens
                q, sc, deq, rel = gptq_linear(dn[e].float(), eye_i if use_eye else Hdn[e], a.damp)
                for k, t in pack_gptq(q.cpu(), sc.cpu()).items(): out[f'{pre}mlp.experts.{e}.down_proj.{k}'] = t
                qdn[e].copy_(deq.to(torch.bfloat16)); report[f'{pre}mlp.experts.{e}.down_proj'] = rel
            del Hdn
        gu.copy_(qgu); dn.copy_(qdn); del qgu, qdn
        nxt = torch.empty_like(hidden)
        with torch.no_grad():
            for ci, s in enumerate(range(0, N * T, a.chunk)):
                y = moe_out(slice(s, s + a.chunk))
                r = refs[ci]; d = ((y.float().cpu() - r) ** 2).sum(1); rr = (r ** 2).sum(1); ho = (torch.arange(s, s + r.shape[0]) >= n_cal)
                ref_sq += rr[~ho].sum().item(); err_sq += d[~ho].sum().item(); ref_ho += rr[ho].sum().item(); err_ho += d[ho].sum().item()
                sh = layer.mlp.shared_expert; xs = x_cpu[s:s + a.chunk].to(dev)
                y = y + torch.sigmoid(layer.mlp.shared_expert_gate(xs)) * sh(xs)
                nxt.view(-1, H)[s:s + a.chunk] = (mid_cpu[s:s + a.chunk].to(dev) + y).to(nxt.dtype)
        hidden = nxt
        shard = f'model-experts-{l:02d}.safetensors'
        save_file(out, f'{a.dst}/{shard}', metadata={'format': 'pt'}); written.append(shard)
        report[f'layer{l}'] = {'moe_relative_output_error': (err_sq / max(ref_sq, 1e-12)) ** 0.5, 'moe_relative_output_error_heldout': (err_ho / max(ref_ho, 1e-12)) ** 0.5 if ref_ho else None, 'rtn_fallback_experts': rtn_experts,
                               'min_tokens_per_expert': int(tokens_per_expert.min()), 'median_tokens_per_expert': int(tokens_per_expert.median())}
        print(f'layer {l} ({layer.block_type}): MoE output rel. error {report[f"layer{l}"]["moe_relative_output_error"]:.4f} held-out {report[f"layer{l}"]["moe_relative_output_error_heldout"]}  rtn experts {rtn_experts}  '
              f'tokens/expert min {int(tokens_per_expert.min())} median {int(tokens_per_expert.median())}  {time.time() - t0:.0f} s', flush=True)
        del layer, out, refs, mid_cpu, x_cpu; torch.cuda.empty_cache()
    # ---- overlay: hard-linked source shards + expert shards + index without the fused expert tensors
    for f in os.listdir(a.src):
        d = f'{a.dst}/{f}'
        if f in written: continue
        if os.path.exists(d): os.remove(d)
        if f.endswith('.safetensors'): os.link(f'{a.src}/{f}', d)
        elif f not in ('model.safetensors.index.json', 'config.json'): shutil.copy(f'{a.src}/{f}', d)
    # fused expert tensors (all layers that were quantised) and the MTP head are not part of the overlay
    done = {f'{P}layers.{l}.mlp.experts.' for l in range(n_layers)}
    wm2 = {k: v for k, v in wm.items() if not k.startswith('mtp.') and not (k.endswith(('.mlp.experts.gate_up_proj', '.mlp.experts.down_proj')) and k.rsplit('.', 1)[0] + '.' in done)}
    for shard in written:
        for k in safe_open(f'{a.dst}/{shard}', framework='pt').keys(): wm2[k] = shard
    ix['weight_map'] = wm2; json.dump(ix, open(f'{a.dst}/model.safetensors.index.json', 'w'))
    c = json.load(open(f'{a.src}/config.json'))
    c['quantization_config'] = {'quant_method': 'gptq', 'bits': 4, 'group_size': G, 'sym': True, 'desc_act': False}
    json.dump(c, open(f'{a.dst}/config.json', 'w'), indent=1)
    json.dump(report, open(f'{a.dst}/conversion_report.json', 'w'))
    print('GPTQ_EXPERTS_DONE', len(written), 'layers', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()

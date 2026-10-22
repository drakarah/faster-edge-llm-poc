"""Layer-by-layer calibrated GPTQ (INT4, group 128, symmetric, MSE-clipped scales) of the DENSE part of Gemma 4 26B-A4B: attention q/k/v/o, dense MLP gate/up and the tied lm_head.

Why a custom loop: the BF16 checkpoint is 51 GB (does not fit RAM or VRAM), so llm-compressor's tools/gptq_g128.py cannot hold the model. Here one decoder layer at a time is built from the safetensors shards on the GPU, the calibration activations (N x seq x hidden, ~0.7 GB) stay on the GPU, Hessians H = X^T X of each linear's input are collected, the layer's linears are quantised with GPTQ (error of each column spread over the remaining ones, tools/gptq_head.py `gptq_rows`), the layer is run again WITH the quantised weights to produce the next layer's inputs (sequential GPTQ), and the layer is freed. MoE experts stay BF16 in the layer for the forward (they are quantised later by the exporter, RTN g64). down_proj (K = 2112) stays BF16. Output = same overlay as tools/dense_int4_overlay_26b.py (hard-linked shards + one INT4 overlay shard + index/config).
Calibration text: generic (nl/en/fr Wikipedia + source code, tools/gptq_g128.py), never the application prompts.
Usage (PoC image, GPU): gptq_dense_layerwise_26b.py SRC_DIR DST_DIR --text-dir /text [--samples 128] [--seq-len 1024] [--layers N]"""
import argparse, json, os, random, shutil, sys, time
from collections import UserDict
import torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ct_to_gptq import pack_gptq, quantize_rows

G = 128
SOURCE_SHARES = {"nl": 0.40, "en": 0.25, "fr": 0.10, "code": 0.25}
WIKI_FILES = {"nl": "20231101.nl-00000-of-00006.parquet", "en": "20231101.en-00000-of-00041.parquet", "fr": "20231101.fr-00000-of-00017.parquet"}


def calibration_ids(text_dir, tokenizer, samples, seq_len, seed=0):
    import pyarrow.parquet as pq
    rng = random.Random(seed); min_chars = seq_len * 5; texts = []
    for source, share in SOURCE_SHARES.items():
        count = round(samples * share)
        if source == "code":
            src = open(os.path.join(text_dir, "code_json.txt"), encoding="utf-8", errors="ignore").read()
            texts += [src[s:s + min_chars] for s in (rng.randrange(0, max(1, len(src) - min_chars)) for _ in range(count))]
        else:
            t = [x for x in pq.read_table(os.path.join(text_dir, WIKI_FILES[source]), columns=["text"]).column("text").to_pylist() if x and len(x) >= min_chars]
            rng.shuffle(t); texts += t[:count]
    rng.shuffle(texts)
    enc = tokenizer(texts[:samples], max_length=seq_len, truncation=True, padding=False)
    rows = [ids for ids in enc["input_ids"] if len(ids) == seq_len]
    print(f"calibration: {len(rows)} sequences x {seq_len} tokens", flush=True)
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def gptq_rows(weight, hinv):
    weight = weight.clone(); rows, cols = weight.shape
    q_all = torch.empty(rows, cols, dtype=torch.int8, device=weight.device); scales = torch.empty(rows, cols // G, device=weight.device)
    for start in range(0, cols, G):
        end = start + G; block = weight[:, start:end].clone(); errors = torch.zeros_like(block)
        _, scale, _ = quantize_rows(block, G); scales[:, start // G] = scale[:, 0]
        hb = hinv[start:end, start:end]
        for i in range(G):
            col = block[:, i]; q = torch.clamp(torch.round(col / scale[:, 0]), -8, 7); q_all[:, start + i] = q.to(torch.int8)
            err = (col - q * scale[:, 0]) / hb[i, i]; block[:, i:] -= err.unsqueeze(1) * hb[i, i:].unsqueeze(0); errors[:, i] = err
        weight[:, end:] -= errors @ hinv[start:end, end:]
    return q_all, scales


@torch.no_grad()
def hinv_factor(H, damp):
    H = H.clone(); dead = torch.diag(H) == 0; H[dead, dead] = 1
    H += damp * torch.diag(H).mean() * torch.eye(H.shape[0], device=H.device)
    L = torch.linalg.cholesky(H); Hinv = torch.cholesky_inverse(L)
    return torch.linalg.cholesky(Hinv, upper=True)


@torch.no_grad()
def gptq_linear(weight, H, damp, chunk=8192, keep_deq=True):
    """Row chunks so a 262k-row head fits next to its Hessian: returns (q, scales, dequantised or None, relative output error)."""
    hinv = hinv_factor(H, damp); qs, ss, ds = [], [], []; err_sum = energy = 0.0
    for s in range(0, weight.shape[0], chunk):
        w = weight[s:s + chunk].float(); q, sc = gptq_rows(w, hinv)
        deq = q.float() * sc.repeat_interleave(G, 1); e = w - deq
        err_sum += ((e @ H) * e).sum().item(); energy += ((w @ H) * w).sum().item()
        qs.append(q.cpu() if not keep_deq else q); ss.append(sc.cpu() if not keep_deq else sc)
        if keep_deq: ds.append(deq)
        del w, deq, e
    q = torch.cat(qs); sc = torch.cat(ss)
    return q, sc, (torch.cat(ds) if keep_deq else None), err_sum / max(energy, 1e-12)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--text-dir', required=True)
    ap.add_argument('--samples', type=int, default=128); ap.add_argument('--seq-len', type=int, default=1024); ap.add_argument('--damp', type=float, default=0.01)
    ap.add_argument('--layers', type=int, default=0, help='only the first N layers (test)'); ap.add_argument('--mb', type=int, default=4); ap.add_argument('--no-lm-head', action='store_true')
    a = ap.parse_args(); dev = 'cuda'
    from transformers import AutoConfig, AutoTokenizer
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.models.gemma4 import modeling_gemma4 as M
    full = AutoConfig.from_pretrained(a.src); cfg = full.text_config; cfg._attn_implementation = 'sdpa'
    tok = AutoTokenizer.from_pretrained(a.src)
    ids = calibration_ids(a.text_dir, tok, a.samples, a.seq_len)
    ix = json.load(open(f'{a.src}/model.safetensors.index.json')); wm = ix['weight_map']; handles = {}
    def get(name):
        sh = wm[name]
        if sh not in handles: handles[sh] = safe_open(f'{a.src}/{sh}', framework='pt')
        return handles[sh].get_tensor(name)
    P = 'model.language_model.'
    emb_w = get(P + 'embed_tokens.weight').to(dev)
    embed = M.Gemma4TextScaledWordEmbedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id, embed_scale=cfg.hidden_size ** 0.5).to(dev, torch.bfloat16)
    embed.weight.data.copy_(emb_w)
    rotary = M.Gemma4TextRotaryEmbedding(cfg).to(dev)
    n_layers = a.layers or cfg.num_hidden_layers
    N, T = ids.shape
    with torch.no_grad():
        hidden = torch.cat([embed(ids[i:i + a.mb].to(dev)) for i in range(0, N, a.mb)])  # [N,T,H] bf16
    pos = torch.arange(T, device=dev).unsqueeze(0)
    out, errs = {}, {}; t0 = time.time()
    for l in range(n_layers):
        layer = M.Gemma4TextDecoderLayer(cfg, l).to(dev, torch.bfloat16).eval()
        pre = f'{P}layers.{l}.'
        sd = {k[len(pre):]: get(k) for k in wm if k.startswith(pre)}
        miss = layer.load_state_dict(sd, strict=False)
        if l == 0: print('missing:', miss.missing_keys[:6], 'unexpected:', miss.unexpected_keys[:6], flush=True)
        targets = {n: m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear) and n in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj')}
        # Hessians per distinct input: q/k/v share one, gate/up share one, o its own
        key_of = {n: ('qkv' if n.startswith('self_attn') and not n.endswith('o_proj') else 'o' if n.endswith('o_proj') else 'gu') for n in targets}
        H = {}
        def make_hook(k):
            def hook(mod, inp):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                if k not in H: H[k] = torch.zeros(x.shape[1], x.shape[1], device=dev)
                H[k] += x.t() @ x
            return hook
        hooks = [m.register_forward_pre_hook(make_hook(key_of[n])) for n, m in targets.items() if n.endswith(('q_proj', 'o_proj', 'gate_proj'))]
        def run(h_in):
            outs = []
            for i in range(0, N, a.mb):
                hs = h_in[i:i + a.mb]; kw = dict(config=cfg, inputs_embeds=hs, attention_mask=None, past_key_values=None, position_ids=pos)
                masks = {'full_attention': create_causal_mask(**kw), 'sliding_attention': create_sliding_window_causal_mask(**kw)}
                lt = cfg.layer_types[l]; pe = rotary(hs, pos, lt)
                with torch.no_grad():
                    outs.append(layer(hs, None, shared_kv_states=UserDict(), position_embeddings=pe, attention_mask=masks[lt], position_ids=pos))
            return torch.cat(outs)
        run(hidden)
        for h in hooks: h.remove()
        for n, m in targets.items():
            q, sc, deq, rel = gptq_linear(m.weight.data, H[key_of[n]], a.damp)
            for k, t in pack_gptq(q.cpu(), sc.cpu()).items(): out[f'{pre}{n}.{k}'] = t
            errs[f'{pre}{n}'] = rel; m.weight.data.copy_(deq.to(m.weight.dtype))
        del H
        hidden = run(hidden)   # next layer's input: from the quantised layer
        print(f'layer {l}: mean output error {sum(v for k, v in errs.items() if k.startswith(pre)) / max(1, len(targets)):.4f}  {time.time() - t0:.0f} s', flush=True)
        del layer, sd, targets; torch.cuda.empty_cache()
    if not a.no_lm_head and n_layers == cfg.num_hidden_layers:
        norm = M.Gemma4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev, torch.bfloat16); norm.weight.data.copy_(get(P + 'norm.weight').to(dev))
        H = torch.zeros(cfg.hidden_size, cfg.hidden_size, device=dev)
        with torch.no_grad():
            for i in range(0, N, a.mb):
                x = norm(hidden[i:i + a.mb]).reshape(-1, cfg.hidden_size).float(); H += x.t() @ x
        q, sc, _, rel = gptq_linear(emb_w, H, a.damp, keep_deq=False)
        for k, t in pack_gptq(q.cpu(), sc.cpu()).items(): out[f'lm_head.{k}'] = t
        errs['lm_head'] = rel; print(f'lm_head output error {rel:.4f}', flush=True)
    # ---- overlay (same layout as dense_int4_overlay_26b.py)
    os.makedirs(a.dst, exist_ok=True)
    for f in os.listdir(a.src):
        d = f'{a.dst}/{f}'
        if os.path.exists(d): os.remove(d)
        if f.endswith('.safetensors'): os.link(f'{a.src}/{f}', d)
        elif f not in ('model.safetensors.index.json', 'config.json'): shutil.copy(f'{a.src}/{f}', d)
    save_file(out, f'{a.dst}/model-dense-int4.safetensors', metadata={'format': 'pt'})
    replaced = {k.rsplit('.', 1)[0] + '.weight' for k in out if k.endswith('.qweight') and not k.startswith('lm_head')}
    wm2 = {k: v for k, v in wm.items() if k not in replaced}
    for k in out: wm2[k] = 'model-dense-int4.safetensors'
    ix['weight_map'] = wm2; json.dump(ix, open(f'{a.dst}/model.safetensors.index.json', 'w'))
    c = json.load(open(f'{a.src}/config.json'))
    c['quantization_config'] = {'quant_method': 'gptq', 'bits': 4, 'group_size': G, 'sym': True, 'desc_act': False, 'lm_head': 'lm_head' in errs}
    if 'lm_head' in errs: c['tie_word_embeddings'] = False; c.get('text_config', {})['tie_word_embeddings'] = False
    json.dump(c, open(f'{a.dst}/config.json', 'w'), indent=1)
    json.dump({'relative_output_error': errs}, open(f'{a.dst}/conversion_report.json', 'w'))
    print('GPTQ_DONE', len(out) // 3, 'linears', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()

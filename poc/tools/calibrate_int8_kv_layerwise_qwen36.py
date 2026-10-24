"""Per-layer INT8 KV-cache scales (EDGELLM_INT8_KV_SCALES JSON) for Qwen3.6-35B-A3B, measured layer by layer (only the 10 full-attention layers own a KV cache).

Same measurement as calibrate_int8_kv_layerwise_26b.py: K amax = largest RoPE pair length of the normed K (rotary dims are rotated pairwise, partial rotary 0.25 of the
256-dim head), V amax of v_proj; scale = amax * margin / 127. One decoder layer at a time is built from the safetensors shards and run on the GPU over the calibration
set (full-precision stream), its output feeds the next layer; the 67 GB checkpoint is never resident. Entries of linear-attention layers are 1.0 (unused).
Usage (PoC image, GPU): calibrate_int8_kv_layerwise_qwen36.py SRC_DIR OUT.json --text-dir /text [--samples 64] [--seq-len 1024] [--margin 1.1]"""
import argparse, json, os, sys, time
import torch
from safetensors import safe_open

for p in ('/tools', os.path.dirname(os.path.abspath(__file__)),
          os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'faster-llm-poc', 'poc', 'tools')):
    if os.path.isdir(p):
        sys.path.insert(0, p)
from gptq_dense_layerwise_26b import calibration_ids  # noqa: E402


def k_pair_amax(k, rotary_dim):
    k = k.float(); half = rotary_dim // 2
    r = torch.sqrt(k[..., :half] ** 2 + k[..., half:rotary_dim] ** 2).amax()
    if rotary_dim < k.shape[-1]:
        r = torch.maximum(r, k[..., rotary_dim:].abs().amax())
    return r.item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--text-dir', required=True)
    ap.add_argument('--samples', type=int, default=64); ap.add_argument('--seq-len', type=int, default=1024); ap.add_argument('--margin', type=float, default=1.1)
    ap.add_argument('--mb', type=int, default=4); ap.add_argument('--layers', type=int, default=0)
    a = ap.parse_args(); dev = 'cuda'
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    cfg = AutoConfig.from_pretrained(a.src).text_config; cfg._attn_implementation = 'sdpa'
    tok = AutoTokenizer.from_pretrained(a.src); ids = calibration_ids(a.text_dir, tok, a.samples, a.seq_len)
    wm = json.load(open(f'{a.src}/model.safetensors.index.json'))['weight_map']; handles = {}
    def get(name):
        sh = wm[name]
        if sh not in handles: handles[sh] = safe_open(f'{a.src}/{sh}', framework='pt')
        return handles[sh].get_tensor(name)
    P_ = 'model.language_model.'; N, T = ids.shape; H_ = cfg.hidden_size
    rope = getattr(cfg, 'rope_parameters', None) or {}; prf = rope.get('partial_rotary_factor', getattr(cfg, 'partial_rotary_factor', 0.25))
    rotary_dim = int(cfg.head_dim * prf)
    rotary = M.Qwen3_5MoeTextRotaryEmbedding(cfg).to(dev); pos4 = torch.arange(T, device=dev).view(1, 1, -1).expand(4, a.mb, -1); text_pos = pos4[0]; rope_pos = pos4[1:]
    emb = torch.nn.Embedding(cfg.vocab_size, H_).to(dev, torch.bfloat16); emb.weight.data.copy_(get(P_ + 'embed_tokens.weight').to(dev))
    hidden = torch.cat([emb(ids[i:i + a.mb].to(dev)) for i in range(0, N, a.mb)]); del emb
    n_layers = a.layers or cfg.num_hidden_layers; k_scale = [1.0] * cfg.num_hidden_layers; v_scale = [1.0] * cfg.num_hidden_layers; t0 = time.time()
    print(f'rotary_dim {rotary_dim} head_dim {cfg.head_dim} samples {N} x {T}', flush=True)
    for l in range(n_layers):
        layer = M.Qwen3_5MoeDecoderLayer(cfg, l).to(dev, torch.bfloat16).eval(); pre = f'{P_}layers.{l}.'
        layer.load_state_dict({k[len(pre):]: get(k) for k in wm if k.startswith(pre)}, strict=False)
        full = layer.block_type == 'full_attention'; kmax = vmax = 0.0
        if full:
            def hk(mod, inp, out):
                nonlocal kmax; kmax = max(kmax, k_pair_amax(out, rotary_dim))
            def hv(mod, inp, out):
                nonlocal vmax; vmax = max(vmax, out.float().abs().amax().item())
            hs_ = [layer.self_attn.k_norm.register_forward_hook(hk), layer.self_attn.v_proj.register_forward_hook(hv)]
        nxt = torch.empty_like(hidden)
        for i in range(0, N, a.mb):
            hs = hidden[i:i + a.mb]; b = hs.shape[0]; cos_sin = rotary(hs, rope_pos[:, :b]); h = layer.input_layernorm(hs)
            if full: h, _ = layer.self_attn(hidden_states=h, attention_mask=None, position_ids=text_pos[:b], past_key_values=None, position_embeddings=cos_sin)
            else: h = layer.linear_attn(hidden_states=h, cache_params=None, attention_mask=None)
            mid = hs + h; nxt[i:i + a.mb] = mid + layer.mlp(layer.post_attention_layernorm(mid))
        hidden = nxt
        if full:
            for h_ in hs_: h_.remove()
            k_scale[l] = kmax * a.margin / 127.0; v_scale[l] = vmax * a.margin / 127.0
            print(f'layer {l} full_attention: K amax {kmax:.3f} V amax {vmax:.3f} -> scales {k_scale[l]:.5f} {v_scale[l]:.5f}  {time.time() - t0:.0f} s', flush=True)
        del layer; torch.cuda.empty_cache()
    json.dump({'k_scale': k_scale, 'v_scale': v_scale, 'margin': a.margin, 'samples': N, 'seq_len': T}, open(a.out, 'w'))
    print('KV_CALIB_DONE', a.out, flush=True)


if __name__ == '__main__':
    main()

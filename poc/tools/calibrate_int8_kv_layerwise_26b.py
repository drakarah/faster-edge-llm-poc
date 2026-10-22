"""Per-layer INT8 KV-cache scales (EDGELLM_INT8_KV_SCALES JSON) for Gemma 4 26B-A4B, measured layer by layer.

Same measurement as tools/calibrate_int8_kv.py (K amax = largest RoPE pair length of the normed K, V amax of the normed V; scale = amax * margin / 127),
but the 51 GB BF16 checkpoint never has to be resident: one decoder layer at a time is built from the safetensors shards (as in
gptq_dense_layerwise_26b.py), run over the calibration set on the GPU and its output becomes the next layer's input. Layers whose global attention
has k == v (attention_k_eq_v) hook v_norm on the raw K projection, which is what the cache stores.

Usage (PoC image, GPU): calibrate_int8_kv_layerwise_26b.py SRC_DIR OUT.json --text-dir /text [--samples 64] [--seq-len 2048] [--margin 1.1]
"""
import argparse, json, os, sys, time
from collections import UserDict
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gptq_dense_layerwise_26b import calibration_ids

INT8_MAX = 127.0


def k_pair_amax(k, rotary_dim):
    k = k.float(); half = rotary_dim // 2
    r = torch.sqrt(k[..., :half] ** 2 + k[..., half:rotary_dim] ** 2).amax()
    if rotary_dim < k.shape[-1]:
        r = torch.maximum(r, k[..., rotary_dim:].abs().amax())
    return r.item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--text-dir', required=True)
    ap.add_argument('--samples', type=int, default=64); ap.add_argument('--seq-len', type=int, default=2048)
    ap.add_argument('--margin', type=float, default=1.1); ap.add_argument('--mb', type=int, default=2)
    ap.add_argument('--layers', type=int, default=0, help='only the first N layers (test)')
    a = ap.parse_args(); dev = 'cuda'
    from transformers import AutoConfig, AutoTokenizer
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.models.gemma4 import modeling_gemma4 as M
    cfg = AutoConfig.from_pretrained(a.src).text_config; cfg._attn_implementation = 'sdpa'
    ids = calibration_ids(a.text_dir, AutoTokenizer.from_pretrained(a.src), a.samples, a.seq_len)
    wm = json.load(open(f'{a.src}/model.safetensors.index.json'))['weight_map']; handles = {}
    def get(name):
        sh = wm[name]
        if sh not in handles: handles[sh] = safe_open(f'{a.src}/{sh}', framework='pt')
        return handles[sh].get_tensor(name)
    P = 'model.language_model.'
    embed = M.Gemma4TextScaledWordEmbedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id, embed_scale=cfg.hidden_size ** 0.5).to(dev, torch.bfloat16)
    embed.weight.data.copy_(get(P + 'embed_tokens.weight').to(dev))
    rotary = M.Gemma4TextRotaryEmbedding(cfg).to(dev)
    N, T = ids.shape
    hidden = torch.cat([embed(ids[i:i + a.mb].to(dev)) for i in range(0, N, a.mb)])
    pos = torch.arange(T, device=dev).unsqueeze(0)
    n_layers = a.layers or cfg.num_hidden_layers
    first_shared = cfg.num_hidden_layers - int(getattr(cfg, 'num_kv_shared_layers', 0) or 0)
    k_amax, v_amax = {}, {}; t0 = time.time()
    for l in range(n_layers):
        layer = M.Gemma4TextDecoderLayer(cfg, l).to(dev, torch.bfloat16).eval()
        pre = f'{P}layers.{l}.'
        layer.load_state_dict({k[len(pre):]: get(k) for k in wm if k.startswith(pre)}, strict=False)
        at = layer.self_attn; hooks = []
        if l < first_shared:
            lt = cfg.layer_types[l]
            factor = float((getattr(cfg, 'rope_parameters', None) or {}).get(lt, {}).get('partial_rotary_factor', 1.0))
            rdim = int(at.head_dim * factor) // 2 * 2
            def kh(_m, _i, out, l=l, rdim=rdim): k_amax[l] = max(k_amax.get(l, 0.0), k_pair_amax(out, rdim))
            def vh(_m, _i, out, l=l): v_amax[l] = max(v_amax.get(l, 0.0), out.float().abs().amax().item())
            hooks = [at.k_norm.register_forward_hook(kh), at.v_norm.register_forward_hook(vh)]
        outs = []
        for i in range(0, N, a.mb):
            hs = hidden[i:i + a.mb]; kw = dict(config=cfg, inputs_embeds=hs, attention_mask=None, past_key_values=None, position_ids=pos)
            masks = {'full_attention': create_causal_mask(**kw), 'sliding_attention': create_sliding_window_causal_mask(**kw)}
            lt = cfg.layer_types[l]
            outs.append(layer(hs, None, shared_kv_states=UserDict(), position_embeddings=rotary(hs, pos, lt), attention_mask=masks[lt], position_ids=pos))
        for h in hooks: h.remove()
        hidden = torch.cat(outs); del layer, outs; torch.cuda.empty_cache()
        print(f'layer {l}: K amax {k_amax.get(l, 0):.2f} V amax {v_amax.get(l, 0):.2f}  {time.time() - t0:.0f} s', flush=True)
    donor = {}
    for l in range(first_shared): donor[cfg.layer_types[l]] = l
    ks, vs = [], []
    for l in range(cfg.num_hidden_layers):
        o = l if l < first_shared else donor[cfg.layer_types[l]]
        if o not in k_amax: raise RuntimeError(f'layer {o}: no statistics (--layers too small?)')
        ks.append(k_amax[o] * a.margin / INT8_MAX); vs.append(v_amax[o] * a.margin / INT8_MAX)
    json.dump({'k_scale': ks, 'v_scale': vs, 'k_amax': k_amax, 'v_amax': v_amax, 'samples': N, 'seq_len': T, 'margin': a.margin}, open(a.out, 'w'), indent=1)
    print(f'KV_CAL_DONE {len(ks)} layers; K amax {min(k_amax.values()):.2f}-{max(k_amax.values()):.2f}, V amax {min(v_amax.values()):.2f}-{max(v_amax.values()):.2f} -> {a.out}', flush=True)


if __name__ == '__main__':
    main()

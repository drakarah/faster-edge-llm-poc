"""Marlin-repacked NVFP4 experts of the shipped checkpoint as a sidecar, for Int4MoePlugin's NVFP4 weight format.
usage: make_nvfp4_sidecar.py SRC_DIR OUT_DIR   (SRC_DIR: original checkpoint dir with model.safetensors.index.json or model.safetensors;
OUT_DIR: merged dir (symlinked shards + index) that receives nvfp4-layerNN.safetensors and the updated index)

Per layer and projection group the plugin takes one global scale: the per-expert multipliers of the shipped tensors differ by 3-5x,
so every expert is re-expressed against the layer maximum (block scales re-rounded to E4M3). MXFP8 layers (no amax tensor) are
dequantized and rounded to NVFP4 first (_round_nvfp4)."""
import sys, os, json, glob
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from tensorrt_edgellm.checkpoint.repacking import repack_nvfp4_a16_marlin_gated_moe_experts

src, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"{src}/config.json")); text = cfg.get("text_config", cfg)
layers, E, inter = int(text["num_hidden_layers"]), int(text["n_routed_experts"]), int(text["intermediate_size"])
padded = ((inter + 127) // 128) * 128
index_path = f"{src}/model.safetensors.index.json"
weight_map = json.load(open(index_path))["weight_map"] if os.path.exists(index_path) else None
handles = {}
def tensor(name):
    f = weight_map[name] if weight_map else "model.safetensors"
    if f not in handles: handles[f] = safe_open(f"{src}/{f}", "pt", device="cpu")
    return handles[f].get_tensor(name)
def has(name): return (name in weight_map) if weight_map else any(name in h.keys() for h in [safe_open(f"{src}/model.safetensors", "pt")])
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
def round_nvfp4(x):
    rows, cols = x.shape
    amax = x.abs().max().clamp_min(1e-30); mult = amax / (6.0 * 448.0)
    blocks = x.view(rows, cols // 16, 16)
    scale = (blocks.abs().amax(-1, keepdim=True) / 6.0 / mult).clamp(max=448.0).to(torch.float8_e4m3fn)
    scaled = (blocks / (scale.float() * mult).clamp_min(1e-30)).abs().clamp(max=6.0)
    index = (scaled.unsqueeze(-1) - E2M1).abs().argmin(-1)
    code = (index + 8 * ((blocks < 0) & (index > 0)).long()).to(torch.uint8).view(rows, cols)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), scale.view(torch.uint8).view(rows, cols // 16), mult
def expert_nvfp4(P, n):
    raw = tensor(P + n).view(torch.uint8); scale = tensor(P + n + "_scale").view(torch.uint8)
    if (P + n + "_amax") in (weight_map or {}) or (weight_map is None and has(P + n + "_amax")):
        mult = tensor(P + n + "_amax").float() / tensor(P + n + "_outer_scale_denominator").float()
        return raw, scale, mult.float()
    e, rows, cols = raw.shape
    packed = torch.empty(e, rows, cols // 2, dtype=torch.uint8); bs = torch.empty(e, rows, cols // 16, dtype=torch.uint8); mult = torch.empty(e)
    for i in range(e):
        dense = (raw[i].view(torch.float8_e4m3fn).float().view(rows, cols // 32, 32) * torch.exp2(scale[i].float() - 127.0)[..., None]).view(rows, cols)
        packed[i], bs[i], mult[i] = round_nvfp4(dense)
    return packed, bs, mult
def unify(scales_u8, mult):
    """Re-express block scales (uint8 E4M3, [E, R, C/16]) against max(mult): scale' = round_e4m3(scale * mult_e / mult_max)."""
    g = mult.max(); ratio = (mult / g).view(-1, 1, 1)
    s = scales_u8.view(torch.float8_e4m3fn).float() * ratio
    return s.clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8), g
os.makedirs(out, exist_ok=True)
single = weight_map is None   # a checkpoint without an index (the tiny test model): one combined file
if not single:
    index = json.load(open(f"{out}/model.safetensors.index.json")); wm = index["weight_map"]
extra = {}
for l in range(layers):
    P = f"model.llm.layers.{l}.mlp.experts."
    w13, s13, m13 = expert_nvfp4(P, "w13_weight"); w2, s2, m2 = expert_nvfp4(P, "w2_weight")
    s13u, g13 = unify(s13, m13); s2u, g2 = unify(s2, m2)
    gp = [w13[i, 0::2] for i in range(E)]; up = [w13[i, 1::2] for i in range(E)]
    gs = [s13u[i, 0::2] for i in range(E)]; us = [s13u[i, 1::2] for i in range(E)]
    one13 = [g13.reshape(1)] * E; one2 = [g2.reshape(1)] * E
    fc1q, fc1s, fc1g, fc2q, fc2s, fc2g = repack_nvfp4_a16_marlin_gated_moe_experts(gp, gs, one13, up, us, one13, [w2[i] for i in range(E)], [s2u[i] for i in range(E)], one2, padded)
    # int32 words: the engine binds the E2M1/E4M3 bytes as INT32 (TensorRT refuses INT8 tensors next to FP8 Q/DQ)
    t = {P + "fc1_qweights": fc1q.contiguous().to(torch.int8).view(torch.int32), P + "fc1_scales": fc1s.contiguous().to(torch.int8).view(torch.int32),
         P + "fc1_global_scales": fc1g.contiguous().to(torch.float16), P + "fc2_qweights": fc2q.contiguous().to(torch.int8).view(torch.int32),
         P + "fc2_scales": fc2s.contiguous().to(torch.int8).view(torch.int32), P + "fc2_global_scales": fc2g.contiguous().to(torch.float16)}
    if single:
        extra.update(t)
    else:
        name = f"nvfp4-layer{l:02d}.safetensors"
        save_file(t, f"{out}/{name}", metadata={"format": "pt"})
        for k in t: wm[k] = name
    print("layer", l, {k.rsplit(".", 1)[1]: tuple(v.shape) for k, v in t.items()}, "g13", float(g13), "g2", float(g2), flush=True)
if single:
    import shutil
    from safetensors.torch import load_file
    allt = load_file(f"{src}/model.safetensors"); allt.update(extra)
    save_file(allt, f"{out}/model.safetensors", metadata={"format": "pt"})
    for f in os.listdir(src):
        if f != "model.safetensors" and os.path.isfile(f"{src}/{f}") and not f.endswith(".pt"): shutil.copy(f"{src}/{f}", f"{out}/{f}")
else:
    json.dump(index, open(f"{out}/model.safetensors.index.json", "w"))
print("DONE")

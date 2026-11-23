"""Add FP8 (e4m3, per-output-row weight scale, static per-tensor input scale) dense attention linears to a merged checkpoint dir.
usage: make_fp8_dense.py AMAX_JSON CKPT_DIR [HF_DIR=/w/hf] [GROUP=32]
Rewrites CKPT_DIR/model.safetensors.index.json to point the dense weights at fp8-dense.safetensors and writes the mixed-precision
hf_quant_config.json (dense linears FP8; the INT4 GPTQ experts stay on the plugin path)."""
import sys, os, json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

amax = json.load(open(sys.argv[1])); out = sys.argv[2]
hf = sys.argv[3] if len(sys.argv) > 3 else "/w/hf"
MAP = {"q_proj.linear": "wq_du", "k_proj.linear": "wk_dv", "v_proj.linear": "wv_dv", "o_proj.linear": "wo_ud",
       "o_proj.gate_proj": "gate_attn", "relative_position_bias.linear": "wr_du"}
index = json.load(open(f"{out}/model.safetensors.index.json")); wm = index["weight_map"]
src_wm = json.load(open(f"{hf}/model.safetensors.index.json"))["weight_map"]   # the original shards hold the bf16 weights
# input scale per original module name (max over everything that shares it)
scales = {}
for k, v in amax.items():
    if k.startswith("lm_head"): continue
    l = int(k.split("layers.")[1].split(".")[0]); rest = k.split("self_attn.")[1].rsplit(".weight", 1)[0]
    scales[f"model.llm.layers.{l}.attn.{MAP[rest]}"] = max(v, 1e-6) / 448.0
if "lm_head.weight" in amax:
    scales["model.llm.unembed"] = max(amax["lm_head.weight"], 1e-6) / 448.0
tensors = {}; layers = {}
for mod, isc in sorted(scales.items()):
    f = safe_open(f"{hf}/{src_wm[mod + '.weight']}", "pt", device="cpu")
    w = f.get_tensor(mod + ".weight").float().cuda()
    ws = (w.abs().amax(-1) / 448.0).clamp_min(1e-12)
    q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device="cuda")
    for i in range(0, w.shape[0], 16384):
        q[i:i + 16384] = (w[i:i + 16384] / ws[i:i + 16384, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    tensors[mod + ".weight"] = q.cpu(); tensors[mod + ".weight_scale"] = ws.half().cpu()
    tensors[mod + ".input_scale"] = torch.tensor(isc, dtype=torch.float32)
    layers[mod] = {"quant_algo": "FP8"}
    del w, q
name = "fp8-dense.safetensors"
save_file(tensors, f"{out}/{name}", metadata={"format": "pt"})
for k in tensors: wm[k] = name
json.dump(index, open(f"{out}/model.safetensors.index.json", "w"))
json.dump({"producer": {"name": "inkling2-fp8-dense"}, "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None, "quantized_layers": layers}},
          open(f"{out}/hf_quant_config.json", "w"), indent=1)
print("fp8 modules", len(layers), "tensors", len(tensors), "file", name)

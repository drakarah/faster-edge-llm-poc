"""DFlash2 block drafter (z-lab/Qwen3.8-27B-DFlash2, bf16) -> INT4 g128 draft checkpoint for the optimised engine.

  python3 dflash2_quantize.py SRC DST

Quantises the q/k/v/o and gate/up/down projections of the draft layers and its fc target projection (5,120 x 25,600) to INT4 with
groups of 128 (symmetric, MSE-clipped: the method of ct_to_gptq.py's lm_head). Convolutions, the candidate selector and the norms
stay as shipped. On the Jetson AGX Orin with Qwen3.8-27B: the draft's GEMMs drop from 23 ms to about 6 ms per round and the weights
from 3.85 GB to about 1.1 GB; accepted tokens per round 3.575 -> 3.565 for the fc projection.
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ct_to_gptq import pack_gptq, quantize_groupwise  # noqa: E402

LAYER = re.compile(r"^layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out, count, error, energy = {}, 0, 0.0, 0.0
    with safe_open(os.path.join(src, "model.safetensors"), "pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if LAYER.match(key) or key == "fc.weight":
                # The fc projection is quantised on the CPU: the measured engine's was, and the clip search is not bit-identical across devices.
                q, scales, relative_error = quantize_groupwise(tensor if key == "fc.weight" else tensor.to(device), 128)
                weight_energy = (tensor.float() ** 2).sum().item()
                error += relative_error * weight_energy
                energy += weight_energy
                count += 1
                for name, value in pack_gptq(q, scales).items():
                    out[key[:-len(".weight")] + "." + name] = value.cpu()
            else:
                out[key] = tensor
    print(f"quantised {count} linears, weighted relative squared error {error / energy:.6f}")
    save_file(out, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    for name in os.listdir(src):
        if name in ("model.safetensors", "config.json"):
            continue
        target = os.path.join(dst, name)
        if not os.path.lexists(target):
            os.symlink(os.path.realpath(os.path.join(src, name)), target)
    config = json.load(open(os.path.join(src, "config.json")))
    config["quantization_config"] = {
        "quant_method": "gptq", "bits": 4, "group_size": 128, "sym": True, "desc_act": False,
        "modules_in_block_to_quantize": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    }
    json.dump(config, open(os.path.join(dst, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

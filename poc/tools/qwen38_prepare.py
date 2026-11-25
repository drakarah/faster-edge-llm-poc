"""Qwen3.8-27B (RedHatAI compressed-tensors INT4) -> the two checkpoints the optimised engine is built from.

  python3 qwen38_prepare.py SRC GPTQ OUT_BF16HEAD OUT_INT4HEAD [--skip-head]

SRC       the downloaded checkpoint (holds model_mtp.safetensors: the bf16 MTP layer, kept bf16: an INT4 MTP MLP gave 1.01 tokens per round)
GPTQ      the output of ct_to_gptq.py (main shard repacked to GPTQ g128, bit-exact)
OUT_*     checkpoint directories: GPTQ main shard + bf16 MTP shard + index; the INT4-head one also carries lm_head.* (INT4 g128,
          MSE-clipped, the same method as ct_to_gptq --quantize-lm-head) and its config marks lm_head as quantised.
The bf16-head export gives the drafter's reduced head (the drafter must not inherit a head sliced from INT4 weights); the INT4-head
export gives the base model. build_fast_qwen38 in lib.sh uses the base of one and the draft of the other.
"""
import json
import os
import sys

from safetensors import safe_open


def link_common(gptq, out, skip=("model.safetensors.index.json", "config.json")):
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(gptq):
        dst = os.path.join(out, name)
        if name in skip or os.path.lexists(dst):
            continue
        os.symlink(os.path.realpath(os.path.join(gptq, name)), dst)


def main_keys(gptq):
    with safe_open(os.path.join(gptq, "model.safetensors"), "pt") as f:
        return list(f.keys())


def write_index(out, weight_map):
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)


def main():
    src, gptq, out_bf16, out_h4 = sys.argv[1:5]
    skip_head = "--skip-head" in sys.argv
    mtp = os.path.realpath(os.path.join(src, "model_mtp.safetensors"))
    with safe_open(mtp, "pt") as f:
        mtp_keys = list(f.keys())
    keys = main_keys(gptq)

    # bf16 head: main shard + bf16 MTP shard
    link_common(gptq, out_bf16)
    if not os.path.lexists(os.path.join(out_bf16, "model_mtp.safetensors")):
        os.symlink(mtp, os.path.join(out_bf16, "model_mtp.safetensors"))
    wm = {k: "model.safetensors" for k in keys}
    wm.update({k: "model_mtp.safetensors" for k in mtp_keys})
    write_index(out_bf16, wm)
    with open(os.path.join(gptq, "config.json")) as f, open(os.path.join(out_bf16, "config.json"), "w") as g:
        g.write(f.read())
    print("bf16-head checkpoint:", out_bf16, len(wm), "tensors")
    if skip_head:
        return

    # INT4 head
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ct_to_gptq import pack_gptq, quantize_groupwise
    from safetensors.torch import save_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with safe_open(os.path.join(gptq, "model.safetensors"), "pt") as f:
        head = f.get_tensor("lm_head.weight").to(device)
    q, scales, err = quantize_groupwise(head, 128)
    print("lm_head relative squared error", err)
    head_tensors = {"lm_head." + k: v.cpu() for k, v in pack_gptq(q, scales).items()}
    link_common(gptq, out_h4)
    save_file(head_tensors, os.path.join(out_h4, "model_head.safetensors"), metadata={"format": "pt"})
    if not os.path.lexists(os.path.join(out_h4, "model_mtp.safetensors")):
        os.symlink(mtp, os.path.join(out_h4, "model_mtp.safetensors"))
    wm = {k: "model.safetensors" for k in keys if k != "lm_head.weight"}
    wm.update({k: "model_mtp.safetensors" for k in mtp_keys})
    wm.update({k: "model_head.safetensors" for k in head_tensors})
    write_index(out_h4, wm)
    with open(os.path.join(gptq, "config.json")) as f:
        cfg = json.load(f)
    qc = cfg["quantization_config"]
    qc["lm_head"] = True
    qc["modules_in_block_to_quantize"] = qc.get("modules_in_block_to_quantize", []) + ["lm_head"]
    with open(os.path.join(out_h4, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("INT4-head checkpoint:", out_h4)


if __name__ == "__main__":
    main()

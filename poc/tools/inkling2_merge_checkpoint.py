"""Merged checkpoint directory for the Inkling 2 builder: links to the original files plus a config the builder reads.

usage: inkling2_merge_checkpoint.py ORIGINAL_DIR MERGED_DIR

Every file of ORIGINAL_DIR is linked into MERGED_DIR (the sidecar tools then add their own files and rewrite the index there),
except config.json, which is copied with two changes:
  - text_config.kv_cache_dtype = "fp8": the engine keeps the FP8 KV cache with unit scales (the checkpoint ships no KV scales);
  - the top-level quantization_config (modelopt NVFP4) is dropped: the quantisation of the dense projections is described by the
    hf_quant_config.json that inkling2_fp8_dense.py writes, and the original entry makes the builder detect the wrong format.
"""
import json
import os
import sys


def merge(original: str, merged: str) -> None:
    os.makedirs(merged, exist_ok=True)
    for name in sorted(os.listdir(original)):
        target = os.path.join(merged, name)
        if name == "config.json" or os.path.lexists(target):
            continue
        os.symlink(os.path.abspath(os.path.join(original, name)), target)
    with open(os.path.join(original, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise SystemExit("config.json has no text_config: not an Inkling 2 checkpoint")
    text["kv_cache_dtype"] = "fp8"
    config.pop("quantization_config", None)
    with open(os.path.join(merged, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    merge(sys.argv[1], sys.argv[2])

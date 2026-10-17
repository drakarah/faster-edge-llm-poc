"""Expand a compressed-tensors INT4 checkpoint (pack-quantized, symmetric) to a dense bf16 one.

Why: calibrated requantisation (GPTQ, gptq_g128.py) needs ordinary bf16 Linear weights to
start from. The QAT weights are dequantised exactly (int4 value x group scale), so the dense
checkpoint holds the very values the QAT model computes with. Every other tensor is copied,
and quantization_config is removed from config.json.

Usage (inside a torch container):
  python3 ct_dequantize.py SRC_DIR DST_DIR
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open

from ct_to_gptq import COPY_FILES, dequantize


SAFETENSORS_DTYPES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int64: "I64",
                      torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL"}


# Write a safetensors file one tensor at a time. `entries` is a list of (name, dtype, shape, produce) with produce()
# returning the tensor; the header is built from the declared dtype and shape first, so peak memory is one tensor
# (the dense 12B checkpoint is 24 GB: a dict of tensors plus save_file's copy does not fit 31 GB of RAM).
def write_streaming(path: str, entries: list) -> None:
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, dtype, shape, _ in entries:
        size = int(torch.tensor([], dtype=dtype).element_size())
        for dim in shape:
            size *= dim
        header[name] = {"dtype": SAFETENSORS_DTYPES[dtype], "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as out:
        out.write(len(blob).to_bytes(8, "little"))
        out.write(blob)
        for name, dtype, shape, produce in entries:
            tensor = produce().contiguous()
            assert tensor.dtype == dtype and list(tensor.shape) == list(shape), name
            out.write(tensor.reshape(-1).view(torch.uint8).numpy().tobytes() if tensor.numel() else b"")


# Write the dense checkpoint: one bf16 .weight per packed linear, everything else unchanged.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    source = safe_open(os.path.join(args.src, "model.safetensors"), "pt")
    keys = list(source.keys())
    packed_bases = {key[: -len(".weight_packed")] for key in keys if key.endswith(".weight_packed")}
    entries = []
    for base in sorted(packed_bases):
        shape = source.get_tensor(base + ".weight_shape").tolist()

        def dense(base=base, shape=shape):
            weight = dequantize(source.get_tensor(base + ".weight_packed"), source.get_tensor(base + ".weight_scale"), shape)
            return weight.to(torch.bfloat16)

        entries.append((base + ".weight", torch.bfloat16, shape, dense))
    for key in keys:
        if key.rsplit(".", 1)[0] in packed_bases and key.rsplit(".", 1)[1] in (
                "weight_packed", "weight_scale", "weight_shape"):
            continue
        piece = source.get_slice(key)
        dtype = {v: k for k, v in SAFETENSORS_DTYPES.items()}[piece.get_dtype()]
        entries.append((key, dtype, piece.get_shape(), lambda key=key: source.get_tensor(key)))
    write_streaming(os.path.join(args.dst, "model.safetensors"), entries)
    tensors = entries

    with open(os.path.join(args.src, "config.json")) as handle:
        config = json.load(handle)
    config.pop("quantization_config", None)
    with open(os.path.join(args.dst, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    for name in COPY_FILES:
        if os.path.exists(os.path.join(args.src, name)):
            shutil.copy(os.path.join(args.src, name), os.path.join(args.dst, name))
    print(f"dequantised {len(packed_bases)} linears, {len(tensors)} tensors -> {args.dst}")


if __name__ == "__main__":
    main()

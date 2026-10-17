"""Quantise Gemma 4's per-layer embedding (PLE) table for the patched Edge-LLM runtime.

Input: an Edge-LLM engine directory with `ple_embedding.safetensors` (one fp16/bf16 tensor
"weight", [vocab, num_layers * ple_hidden]). Output: a new engine directory whose files are
hard links to the input's, except `ple_embedding.safetensors`, which holds
  weight  int8 [vocab, num_layers * ple_hidden]   symmetric, value = q * scale
  scale   fp16 [vocab, num_layers]                one scale per (token, layer) slice
and is read by the fork's Gemma4EmbeddingPreprocessor (INT8 PLE gather).

With --token-embedding, `embedding.safetensors` is also quantised: "embedding" int8
[vocab, hidden] + "embedding_scale" fp32 [vocab, hidden / 128] (the layout of Edge-LLM's FP8
embedding, read through the same INT8 loader).

The tables are only gathered (42 × 256 values per token), so its precision costs memory, not
decode speed. Prints the relative squared reconstruction error.

Usage (inside a torch container):
  python3 quantize_ple.py SRC_ENGINE_DIR DST_ENGINE_DIR [--ple-hidden 256] [--chunk-rows 16384]
"""

import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PLE_FILE = "ple_embedding.safetensors"
EMBEDDING_FILE = "embedding.safetensors"
EMBEDDING_GROUP = 128  # must match kFP8EmbeddingBlockSize in Edge-LLM (shared INT8/FP8 layout)


# Quantise rows [n, layers * hidden] to int8 with one scale per hidden-sized slice.
def quantize_rows(rows: torch.Tensor, hidden: int) -> tuple:
    slices = rows.float().reshape(rows.shape[0], -1, hidden)
    scale = (slices.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-8)
    q = torch.clamp(torch.round(slices / scale), -127, 127).to(torch.int8)
    error = ((q.float() * scale - slices) ** 2).sum().item()
    energy = (slices ** 2).sum().item()
    return q.reshape(rows.shape[0], -1), scale.squeeze(-1).to(torch.float16), error, energy


# Hard-link every file of SRC into DST except the PLE table, which is written quantised.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--ple-hidden", type=int, default=256)
    parser.add_argument("--chunk-rows", type=int, default=16384)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--token-embedding", action="store_true", help="also quantise embedding.safetensors")
    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    for name in os.listdir(args.src):
        source, target = os.path.join(args.src, name), os.path.join(args.dst, name)
        if name == PLE_FILE or os.path.isdir(source) or (args.token_embedding and name == EMBEDDING_FILE):
            continue
        if os.path.exists(target):
            os.remove(target)
        os.link(source, target)

    with safe_open(os.path.join(args.src, PLE_FILE), framework="pt") as handle:
        table = handle.get_slice("weight")
        vocab, width = table.get_shape()
        q_parts, scale_parts, error_sum, energy_sum = [], [], 0.0, 0.0
        for start in range(0, vocab, args.chunk_rows):
            rows = table[start:start + args.chunk_rows].to(args.device)
            q, scale, error, energy = quantize_rows(rows, args.ple_hidden)
            q_parts.append(q.cpu())
            scale_parts.append(scale.cpu())
            error_sum += error
            energy_sum += energy
            del rows, q, scale
    save_file({"weight": torch.cat(q_parts), "scale": torch.cat(scale_parts)},
              os.path.join(args.dst, PLE_FILE), metadata={"format": "pt"})
    summary = {"vocab": vocab, "width": width, "slices_per_row": width // args.ple_hidden,
               "relative_sq_error": error_sum / energy_sum,
               "bytes": os.path.getsize(os.path.join(args.dst, PLE_FILE))}
    if args.token_embedding:
        summary["token_embedding"] = quantize_token_embedding(args)
    print(json.dumps(summary))


# Quantise embedding.safetensors to int8 + fp32 group scales; returns a summary dict.
def quantize_token_embedding(args) -> dict:
    with safe_open(os.path.join(args.src, EMBEDDING_FILE), framework="pt") as handle:
        table = handle.get_slice("embedding")
        vocab, hidden = table.get_shape()
        q_parts, scale_parts, error_sum, energy_sum = [], [], 0.0, 0.0
        for start in range(0, vocab, args.chunk_rows):
            rows = table[start:start + args.chunk_rows].to(args.device)
            q, scale, error, energy = quantize_rows(rows, EMBEDDING_GROUP)
            q_parts.append(q.cpu())
            scale_parts.append(scale.float().cpu())
            error_sum += error
            energy_sum += energy
    save_file({"embedding": torch.cat(q_parts), "embedding_scale": torch.cat(scale_parts)},
              os.path.join(args.dst, EMBEDDING_FILE), metadata={"format": "pt"})
    return {"vocab": vocab, "hidden": hidden, "relative_sq_error": error_sum / energy_sum,
            "bytes": os.path.getsize(os.path.join(args.dst, EMBEDDING_FILE))}


if __name__ == "__main__":
    main()

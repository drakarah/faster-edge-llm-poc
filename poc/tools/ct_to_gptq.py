"""Convert a compressed-tensors INT4 checkpoint (pack-quantized, symmetric, any group size)
into a GPTQ-layout INT4 checkpoint with a different group size (default 128).

Why: Google ships Gemma 4 QAT weights as compressed-tensors W4A16 with group size 32, and
TensorRT Edge-LLM's Orin INT4 kernels accept only GPTQ/AWQ layouts with group size 128. The
QAT weights are dequantised exactly, then requantised per 128-wide group with an MSE-optimal
clipping search (not plain max-abs), which keeps the error of the coarser grid low.

When the source already uses the target group size (e.g. the GPTQ output of gptq_g128.py), its
integers and scales are repacked as they are, bit-exactly; re-rounding would undo GPTQ's choices.
Note: the RTN path (different group size) cost clearly more quality than calibrated GPTQ; prefer ct_dequantize.py + gptq_g128.py and use this for repacking.
Every other tensor (embeddings, norms, vision/audio towers) is copied unchanged. A stored
lm_head.weight is dropped when the config ties it to embed_tokens (runtimes ignore it). With
--quantize-lm-head the head the runtimes use (embed_tokens when tied) is quantised to INT4
with the same method, stored as lm_head.*, and the config is untied (embed_tokens stays bf16
as the input lookup table).

Output layout per quantised linear (GPTQ, sym, desc_act=False):
  qweight [in/8, out] int32 (8 nibbles packed along the input axis), scales [in/G, out] fp16,
  qzeros [in/G, out/8] int32 with every nibble 8 (real zero point 8 -> Edge-LLM offset 0).

Usage (inside a torch container):
  python3 ct_to_gptq.py SRC_DIR DST_DIR [--group-size 128] [--quantize-lm-head] [--device cuda]
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CLIP_RATIOS = [1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76]
QUANTIZED_MODULE_NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "per_layer_input_gate",
                          "per_layer_projection", "per_layer_model_projection"]
COPY_FILES = ["chat_template.jinja", "generation_config.json", "processor_config.json",
              "tokenizer.json", "tokenizer_config.json"]


# Unpack a compressed-tensors pack-quantized int32 tensor [out, in * bits / 32] to signed integers [out, in]. 4-bit (the default) packs
# 8 values per word; other widths (3-bit: INT3 experiment) are an LSB-first bit stream, so the width follows from the packed shape.
def unpack_compressed_tensors(packed: torch.Tensor, in_features: int | None = None) -> torch.Tensor:
    bits = 4 if in_features is None else round(packed.shape[1] * 32 / in_features)
    if bits == 4:
        shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
        nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF  # [out, in/8, 8]
        return nibbles.reshape(packed.shape[0], -1).to(torch.int8) - 8
    words = packed.to(torch.int64) & 0xFFFFFFFF
    words = torch.cat([words, torch.zeros_like(words[:, :1])], dim=1)
    offset = torch.arange(in_features, device=packed.device) * bits
    low, shift = offset // 32, offset % 32
    value = ((words[:, low] >> shift) | (words[:, low + 1] << (32 - shift))) & ((1 << bits) - 1)
    return value.to(torch.int8) - (1 << (bits - 1))


# Dequantise one compressed-tensors linear to float32 [out, in].
def dequantize(packed: torch.Tensor, scale: torch.Tensor, shape: list[int]) -> torch.Tensor:
    values = unpack_compressed_tensors(packed, shape[1])[:, : shape[1]].float()
    group = shape[1] // scale.shape[1]
    return values * scale.float().repeat_interleave(group, dim=1)


# Quantise in chunks of output rows so large tensors (a 262k-row lm_head) stay within memory;
# returns the same (q, scales, relative squared error) as quantize_rows.
def quantize_groupwise(weight: torch.Tensor, group_size: int, chunk_rows: int = 8192) -> tuple:
    q_parts, scale_parts, error_sum, energy_sum = [], [], 0.0, 0.0
    for start in range(0, weight.shape[0], chunk_rows):
        rows = weight[start:start + chunk_rows].float()
        q, scales, relative_error = quantize_rows(rows, group_size)
        energy = (rows ** 2).sum().item()
        error_sum += relative_error * energy
        energy_sum += energy
        q_parts.append(q)
        scale_parts.append(scales)
        del rows
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(q_parts), torch.cat(scale_parts), error_sum / max(energy_sum, 1e-12)


# Symmetric INT4 quantisation per group with an MSE-optimal clip; returns (q [out,in] int8 in
# [-8,7], scales [out, in/G] float32, relative squared error).
def quantize_rows(weight: torch.Tensor, group_size: int) -> tuple:
    out_features, in_features = weight.shape
    groups = weight.reshape(out_features, in_features // group_size, group_size)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    best_error = torch.full(amax.shape, float("inf"), device=weight.device)
    best_scale = amax / 7
    for ratio in CLIP_RATIOS:
        scale = amax * ratio / 7
        q = torch.clamp(torch.round(groups / scale), -8, 7)
        error = ((q * scale - groups) ** 2).sum(dim=-1, keepdim=True)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
    q = torch.clamp(torch.round(groups / best_scale), -8, 7).to(torch.int8)
    relative_error = (best_error.sum() / (groups ** 2).sum().clamp(min=1e-12)).item()
    return q.reshape(out_features, in_features), best_scale.squeeze(-1), relative_error


# Pack signed int4 [out, in] into GPTQ tensors (qweight, qzeros, scales fp16).
def pack_gptq(q: torch.Tensor, scales: torch.Tensor) -> dict:
    out_features, in_features = q.shape
    unsigned = (q.to(torch.int64) + 8).t().contiguous()  # [in, out]
    unsigned = unsigned.reshape(in_features // 8, 8, out_features)
    shifts = torch.arange(0, 32, 4, device=q.device, dtype=torch.int64).view(1, 8, 1)
    qweight = (unsigned << shifts).sum(dim=1)
    qweight = ((qweight + 2**31) % 2**32 - 2**31).to(torch.int32)  # wrap to signed int32
    num_groups = scales.shape[1]
    zero_word = torch.tensor(0x88888888 - 2**32, dtype=torch.int32)
    qzeros = zero_word.repeat(num_groups, out_features // 8)
    return {"qweight": qweight.cpu(), "qzeros": qzeros.cpu(),
            "scales": scales.t().contiguous().to(torch.float16).cpu()}


# Convert every linear of SRC to GPTQ in DST and write the new config and side files.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--quantize-lm-head", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    with open(os.path.join(args.src, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    tied = bool(config.get("tie_word_embeddings", False))

    tensors_out: dict[str, torch.Tensor] = {}
    errors: dict[str, float] = {}
    with safe_open(os.path.join(args.src, "model.safetensors"), framework="pt") as source:
        keys = list(source.keys())
        packed_modules = sorted({k[: -len(".weight_packed")] for k in keys if k.endswith(".weight_packed")})
        handled = set()
        for module in packed_modules:
            packed = source.get_tensor(module + ".weight_packed").to(args.device)
            scale = source.get_tensor(module + ".weight_scale").to(args.device)
            shape = source.get_tensor(module + ".weight_shape").tolist()
            if shape[1] % args.group_size:
                raise ValueError(f"{module}: in_features {shape[1]} not divisible by {args.group_size}")
            if shape[1] // scale.shape[1] == args.group_size:
                # Already on the target grid (e.g. GPTQ output from gptq_g128.py): repack
                # the integers and scales as they are. Re-rounding would undo GPTQ's choices.
                q = unpack_compressed_tensors(packed, shape[1])[:, : shape[1]]
                scales = scale.float()
                errors[module] = 0.0
            else:
                weight = dequantize(packed, scale, shape)
                q, scales, errors[module] = quantize_groupwise(weight, args.group_size)
                del weight
            for name, tensor in pack_gptq(q, scales).items():
                tensors_out[f"{module}.{name}"] = tensor
            del packed, scale, q, scales
            handled.update({module + suffix for suffix in (".weight_packed", ".weight_scale", ".weight_shape")})
        for key in keys:
            if key in handled:
                continue
            if key == "lm_head.weight" and (tied or args.quantize_lm_head):
                continue  # tied: runtimes reuse embed_tokens, so this copy is dead weight
            tensors_out[key] = source.get_tensor(key)
        if args.quantize_lm_head:
            # The head the runtimes actually use: embed_tokens when tied, else lm_head.weight.
            head_key = "model.embed_tokens.weight" if tied else "lm_head.weight"
            head = source.get_tensor(head_key).to(args.device)  # bf16; chunks go to fp32
            q, scales, errors["lm_head"] = quantize_groupwise(head, args.group_size)
            for name, packed_tensor in pack_gptq(q, scales).items():
                tensors_out[f"lm_head.{name}"] = packed_tensor

    save_file(tensors_out, os.path.join(args.dst, "model.safetensors"), metadata={"format": "pt"})

    config["quantization_config"] = {"quant_method": "gptq", "bits": 4, "group_size": args.group_size,
                                     "sym": True, "desc_act": False, "lm_head": args.quantize_lm_head,
                                     # Explicit list: vLLM otherwise infers quantised modules from
                                     # the tensors present, and Gemma 4's KV-sharing layers have no
                                     # k_proj/v_proj, which makes its fused qkv_proj look mixed.
                                     "modules_in_block_to_quantize": QUANTIZED_MODULE_NAMES
                                     + (["lm_head"] if args.quantize_lm_head else [])}
    if args.quantize_lm_head:
        config["tie_word_embeddings"] = False  # the head is now its own INT4 tensor
    with open(os.path.join(args.dst, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    for name in COPY_FILES:
        if os.path.exists(os.path.join(args.src, name)):
            shutil.copy(os.path.join(args.src, name), os.path.join(args.dst, name))

    ranked = sorted(errors.items(), key=lambda item: -item[1])
    summary = {"modules": len(errors), "group_size": args.group_size,
               "mean_relative_sq_error": sum(errors.values()) / len(errors),
               "worst": ranked[:5]}
    with open(os.path.join(args.dst, "conversion_report.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "per_module": errors}, handle, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

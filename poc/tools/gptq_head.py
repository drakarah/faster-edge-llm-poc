"""Calibrated GPTQ INT4 (group 128) for Gemma 4's output head only, added to a GPTQ checkpoint.

Why: the bf16 262k-row head (1.34 GB) is the largest read per decoded token. Edge-LLM's fast
INT4 GEMV takes a GPTQ head, but round-to-nearest INT4 (ct_to_gptq.py --quantize-lm-head)
cost answers in an application-level eval, and so does a reduced vocabulary. GPTQ spreads
each column's rounding error over the remaining columns, weighted by the head's real input
statistics H = X^T X (final hidden states), so the logits stay close where single weights move.

Only the head is quantised here, so H comes from the already-quantised body (the compressed-tensors
GPTQ checkpoint; it is what feeds the head at serving time). Calibration text is the same generic
mix as gptq_g128.py (never the benchmark's prompts). Rows of the head are independent given H, so
they are processed in chunks. Prints the head's relative output error tr(E H E^T) / tr(W H W^T)
for RTN and GPTQ.

Usage (llm-compressor venv, inside the poc container):
  python3 gptq_head.py CT_DIR GPTQ_DIR DST_DIR --text-dir /work/text [--samples 128] [--seq-len 1024]
CT_DIR: compressed-tensors checkpoint (for the forward pass); GPTQ_DIR: its GPTQ repack
(ct_to_gptq.py), copied to DST_DIR with lm_head.{qweight,qzeros,scales} added and untied.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

from ct_to_gptq import pack_gptq, quantize_groupwise, quantize_rows
from gptq_g128 import calibration_dataset

GROUP = 128


# Runs an embedding lookup on the host and returns the result on the GPU. The wrapped module is kept
# in a plain list so Module.to("cuda") does not move its table.
class _HostEmbedding(torch.nn.Module):
    def __init__(self, embedding: torch.nn.Module):
        super().__init__()
        self._host = [embedding]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._host[0](input_ids.cpu()).to("cuda")


# Accumulate H = sum X^T X over the final hidden states (the head's input), in float32.
@torch.no_grad()
def head_hessian(model, dataset) -> torch.Tensor:
    hidden = model.config.get_text_config().hidden_size
    hessian = torch.zeros(hidden, hidden, dtype=torch.float32, device="cuda")
    base = model.model
    for index, row in enumerate(dataset):
        ids = torch.tensor([row["input_ids"]], device="cuda")
        states = base(input_ids=ids).last_hidden_state[0].float()  # [T, hidden], after final norm
        hessian += states.t() @ states
        if index % 16 == 0:
            print(f"  hessian: {index + 1}/{len(dataset)}", flush=True)
    return hessian


# GPTQ (no act-order) of rows [n, hidden] given the inverse-Hessian Cholesky factor.
@torch.no_grad()
def gptq_rows(weight: torch.Tensor, hinv: torch.Tensor) -> tuple:
    weight = weight.clone()
    rows, cols = weight.shape
    q_all = torch.empty(rows, cols, dtype=torch.int8, device=weight.device)
    scales = torch.empty(rows, cols // GROUP, device=weight.device)
    for start in range(0, cols, GROUP):
        end = start + GROUP
        block = weight[:, start:end].clone()
        errors = torch.zeros_like(block)
        _, scale, _ = quantize_rows(block, GROUP)  # MSE-clipped scale of the updated group
        scales[:, start // GROUP] = scale[:, 0]
        hblock = hinv[start:end, start:end]
        for i in range(GROUP):
            column = block[:, i]
            q = torch.clamp(torch.round(column / scale[:, 0]), -8, 7)
            q_all[:, start + i] = q.to(torch.int8)
            err = (column - q * scale[:, 0]) / hblock[i, i]
            block[:, i:] -= err.unsqueeze(1) * hblock[i, i:].unsqueeze(0)
            errors[:, i] = err
        weight[:, end:] -= errors @ hinv[start:end, end:]
    return q_all, scales


# Relative output error of a quantised head: tr(E H E^T) / tr(W H W^T), in row chunks.
@torch.no_grad()
def output_error(weight: torch.Tensor, q: torch.Tensor, scales: torch.Tensor, hessian: torch.Tensor) -> float:
    err_sum, energy = 0.0, 0.0
    for start in range(0, weight.shape[0], 16384):
        w = weight[start:start + 16384].float().cuda()
        deq = q[start:start + 16384].cuda().float() * scales[start:start + 16384].cuda().repeat_interleave(GROUP, 1)
        e = w - deq
        err_sum += ((e @ hessian) * e).sum().item()
        energy += ((w @ hessian) * w).sum().item()
    return err_sum / energy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ct_dir")
    parser.add_argument("gptq_dir")
    parser.add_argument("dst")
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--chunk-rows", type=int, default=32768)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.ct_dir)
    # CT_DIR may also be a dense bf16 expansion of the compressed checkpoint (ct_dequantize.py):
    # the loader keeps compressed modules offloaded on the host, so .to("cuda") below would not move them.
    quant = {} if not json.load(open(os.path.join(args.ct_dir, "config.json"))).get("quantization_config") else {
        # dense weights: the compressed modules have no .weight
        "quantization_config": CompressedTensorsConfig(run_compressed=False)}
    model = AutoModelForCausalLM.from_pretrained(args.ct_dir, dtype="bfloat16", device_map=None, **quant)
    model.eval()
    # An 8 GB card cannot hold the dense bf16 model (8 GB of layers + a 5.25 GiB per-layer embedding
    # table). The table stays on the host (a lookup, cheap on the CPU); each decoder layer is moved to
    # the GPU for its forward pass only; everything else lives on the GPU.
    if hasattr(model.model, "embed_tokens_per_layer"):  # E-models only; the 12B has no per-layer table
        model.model.embed_tokens_per_layer = _HostEmbedding(model.model.embed_tokens_per_layer)
    for name, child in model.named_children():
        if name != "model":
            child.to("cuda")
    for name, child in model.model.named_children():
        if name != "layers":
            child.to("cuda")
    for layer in model.model.layers:
        layer.register_forward_pre_hook(lambda module, args: (module.to("cuda"), None)[1])
        layer.register_forward_hook(lambda module, args, output: (module.to("cpu"), None)[1])
    head = model.get_output_embeddings().weight.detach().float().cpu()  # tied: embed_tokens
    dataset = calibration_dataset(args.text_dir, tokenizer, args.samples, args.seq_len, seed=1)
    cache = args.dst.rstrip("/") + ".hessian.pt"
    if os.path.exists(cache):
        hessian = torch.load(cache).cuda()
    else:
        hessian = head_hessian(model, dataset)
        torch.save(hessian.cpu(), cache)
    del model
    torch.cuda.empty_cache()

    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    head[:, dead.cpu()] = 0
    damped = hessian + args.damp * torch.mean(torch.diag(hessian)) * torch.eye(len(hessian), device="cuda")
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(damped)), upper=True)

    q_parts, scale_parts = [], []
    for start in range(0, head.shape[0], args.chunk_rows):
        q, scales = gptq_rows(head[start:start + args.chunk_rows].cuda(), hinv)
        q_parts.append(q.cpu())
        scale_parts.append(scales.cpu())
        print(f"  gptq rows {start + q.shape[0]}/{head.shape[0]}", flush=True)
    q_gptq, s_gptq = torch.cat(q_parts), torch.cat(scale_parts)
    # RTN reference for the report, in row chunks: the fp32 head of a 262k x 3 840 model (4 GB) next to the Hessian does not fit an 8 GB card.
    rtn = [quantize_groupwise(head[start:start + 16384].cuda(), GROUP) for start in range(0, head.shape[0], 16384)]
    q_rtn, s_rtn = torch.cat([part[0].cpu() for part in rtn]), torch.cat([part[1].cpu() for part in rtn])
    report = {"rtn_rel_output_error": output_error(head, q_rtn, s_rtn, hessian),
              "gptq_rel_output_error": output_error(head, q_gptq, s_gptq, hessian),
              "samples": len(dataset), "seq_len": args.seq_len, "damp": args.damp}
    print(json.dumps(report, indent=1))

    os.makedirs(args.dst, exist_ok=True)
    for name in os.listdir(args.gptq_dir):
        if name not in ("model.safetensors", "config.json"):
            src = os.path.join(args.gptq_dir, name)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(args.dst, name))
    tensors = {}
    with safe_open(os.path.join(args.gptq_dir, "model.safetensors"), framework="pt") as source:
        for key in source.keys():
            if key != "lm_head.weight":
                tensors[key] = source.get_tensor(key)
    for name, tensor in pack_gptq(q_gptq, s_gptq).items():
        tensors[f"lm_head.{name}"] = tensor
    save_file(tensors, os.path.join(args.dst, "model.safetensors"), metadata={"format": "pt"})

    with open(os.path.join(args.gptq_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    quant = config["quantization_config"]
    quant["lm_head"] = True
    quant["modules_in_block_to_quantize"] = sorted(set(quant["modules_in_block_to_quantize"]) | {"lm_head"})
    config["tie_word_embeddings"] = False
    if "text_config" in config:
        config["text_config"]["tie_word_embeddings"] = False
    with open(os.path.join(args.dst, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with open(os.path.join(args.dst, "head_gptq_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print(f"saved -> {args.dst}")


if __name__ == "__main__":
    main()

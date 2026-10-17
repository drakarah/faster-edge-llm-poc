"""Calibrated GPTQ requantisation of a dense Gemma 4 checkpoint to INT4, group size 128.

Why: TensorRT Edge-LLM's Orin INT4 kernels accept group size 128 only, while Google's QAT
checkpoint is group 32. Rounding the QAT weights onto a 128-wide grid (ct_to_gptq.py,
MSE clip search per group) cost 12 of 43 regressed tests in an application-level eval.
GPTQ instead quantises each weight column in turn and spreads its rounding error over the
columns not yet quantised, weighted by real activations (the Hessian X^T X of each layer's
input), so the layer's OUTPUT stays close even where single weights move.

Input is the dense checkpoint from ct_dequantize.py (the exact QAT values). Calibration
text is generic: Dutch, English and French Wikipedia plus source code, never the benchmark's
own prompts (the benchmark is the measurement and must not be what the weights were tuned on).
Every Linear except lm_head is quantised (same targets as Google's QAT config). The model is
processed one decoder layer at a time (llm-compressor's sequential pipeline), so only one
layer's Hessians are held at once. Output is compressed-tensors (pack-quantized, group 128);
vLLM loads it directly and ct_to_gptq.py repacks it for Edge-LLM.

Usage (venv with llm-compressor on top of the container's torch):
  python3 gptq_g128.py SRC_DENSE_DIR DST_DIR --text-dir /work/text [--samples 256] [--seq-len 1024]
"""
import argparse
import os
import random

import pyarrow.parquet as pq
from datasets import Dataset
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import base as gptq_base
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

_compress_module_list = gptq_base.GPTQModifier.compress_module_list


# llm-compressor 0.14 fills a module's weight observer per decoder layer, so a Linear outside
# the decoder layers (Gemma 4's model.per_layer_model_projection, which runs before layer 0)
# reaches compression with Hessian statistics but no weight statistics and fails with "No
# statistics available". Observe such weights first; GPTQ then treats them like any other.
def _compress_module_list_observing_weights(self, module_list):
    for module in module_list:
        if not module.weight_observer.has_statistics:
            module.weight_observer(module.weight)
    return _compress_module_list(self, module_list)


gptq_base.GPTQModifier.compress_module_list = _compress_module_list_observing_weights


# Gemma 4's KV-sharing layers read the keys/values of an earlier layer from a dict that the
# model passes to every layer and the storing layers fill as a side effect. The sequential
# pipeline runs one decoder layer at a time from cached inputs, so that side effect never
# reaches the later layers ("KeyError: 'sliding_attention'" at layer 24). Keep one shared-KV
# dict per calibration sample instead: with batch size 1, every layer sees the samples in the
# same fixed order, in the calibration pass and again in the propagation pass, so an attention
# module's call count modulo the sample count identifies the sample. The storing layers run to
# completion (both passes) before the first sharing layer, so readers see the propagated values.
_SAMPLE_KV_STATES: list[dict] = []


def _share_kv_per_sample(num_samples: int) -> None:
    from transformers.models.gemma4 import modeling_gemma4

    _SAMPLE_KV_STATES[:] = [{} for _ in range(num_samples)]
    original_forward = modeling_gemma4.Gemma4TextAttention.forward

    def forward(self, hidden_states, position_embeddings, attention_mask, shared_kv_states,
                *args, **kwargs):
        calls = getattr(self, "_calibration_calls", 0)
        self._calibration_calls = calls + 1
        sample_states = _SAMPLE_KV_STATES[calls % num_samples]
        return original_forward(self, hidden_states, position_embeddings, attention_mask,
                                sample_states, *args, **kwargs)

    modeling_gemma4.Gemma4TextAttention.forward = forward

# Share of calibration samples per source; Dutch first (the recorded benchmark workload is Dutch).
SOURCE_SHARES = {"nl": 0.40, "en": 0.25, "fr": 0.10, "code": 0.25}
WIKI_FILES = {"nl": "20231101.nl-00000-of-00006.parquet",
              "en": "20231101.en-00000-of-00041.parquet",
              "fr": "20231101.fr-00000-of-00017.parquet"}


# Collect `count` article texts of at least `min_chars` from one Wikipedia parquet file.
def wiki_texts(path: str, count: int, min_chars: int, rng: random.Random) -> list[str]:
    texts = [t for t in pq.read_table(path, columns=["text"]).column("text").to_pylist()
             if t and len(t) >= min_chars]
    rng.shuffle(texts)
    return texts[:count]


# Cut a long source-code file into `count` chunks of `chunk_chars` from random offsets.
def code_texts(path: str, count: int, chunk_chars: int, rng: random.Random) -> list[str]:
    with open(path, encoding="utf-8", errors="ignore") as handle:
        source = handle.read()
    starts = [rng.randrange(0, max(1, len(source) - chunk_chars)) for _ in range(count)]
    return [source[start:start + chunk_chars] for start in starts]


# Build the tokenised calibration set: `samples` sequences of exactly `seq_len` tokens.
def calibration_dataset(text_dir: str, tokenizer, samples: int, seq_len: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    min_chars = seq_len * 5  # comfortably more than seq_len tokens of prose
    texts: list[str] = []
    for source, share in SOURCE_SHARES.items():
        count = round(samples * share)
        if source == "code":
            texts += code_texts(os.path.join(text_dir, "code_json.txt"), count, min_chars, rng)
        else:
            texts += wiki_texts(os.path.join(text_dir, WIKI_FILES[source]), count, min_chars, rng)
    rng.shuffle(texts)
    encoded = tokenizer(texts[:samples], max_length=seq_len, truncation=True, padding=False)
    rows = [(ids, mask) for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
            if len(ids) == seq_len]
    print(f"calibration: {len(rows)} sequences x {seq_len} tokens")
    return Dataset.from_dict({"input_ids": [r[0] for r in rows],
                              "attention_mask": [r[1] for r in rows]})


# Run GPTQ (W4A16, symmetric, group 128) over every Linear except lm_head and save compressed.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ffn-bits", type=int, default=4, help="bit width of the MLP projections (3 = INT3 experiment, IDEAS)")
    parser.add_argument("--attn-bits", type=int, default=4, help="bit width of the attention projections")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.src)
    model = AutoModelForCausalLM.from_pretrained(args.src, dtype="bfloat16")
    dataset = calibration_dataset(args.text_dir, tokenizer, args.samples, args.seq_len, args.seed)

    # Gemma 4 E-models share KV between layers (patch needed); the 12B "unified" family does not.
    if getattr(model.config.get_text_config(), "num_kv_shared_layers", 0):
        _share_kv_per_sample(len(dataset))
    layer_classes = sorted({type(module).__name__ for module in model.modules()
                            if type(module).__name__.endswith("DecoderLayer")})
    print("sequential targets:", layer_classes)

    if args.ffn_bits == 4 and args.attn_bits == 4:
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"],
                              dampening_frac=0.01)
    else:  # mixed widths: two config groups (MLP projections / everything else), symmetric group-128 each
        def group(targets, bits):
            return QuantizationScheme(targets=targets, weights=QuantizationArgs(
                num_bits=bits, type="int", symmetric=True, strategy="group", group_size=128))
        recipe = GPTQModifier(config_groups={
            "mlp": group(["re:.*mlp\\.(gate|up|down)_proj$"], args.ffn_bits),
            "rest": group(["re:.*(self_attn\\.(q|k|v|o)_proj|per_layer_input_gate|per_layer_projection|per_layer_model_projection)$"], args.attn_bits)},
            ignore=["lm_head"], dampening_frac=0.01)
    oneshot(model=model, processor=tokenizer, dataset=dataset, recipe=recipe,
            max_seq_length=args.seq_len, num_calibration_samples=len(dataset),
            batch_size=1, shuffle_calibration_samples=False,
            sequential_targets=layer_classes)

    model.save_pretrained(args.dst, save_compressed=True)
    tokenizer.save_pretrained(args.dst)
    print(f"saved -> {args.dst}")


if __name__ == "__main__":
    main()

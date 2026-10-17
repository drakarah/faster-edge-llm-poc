"""Strip the vision and audio towers from a Gemma 4 checkpoint, leaving a text-only model.

Keeps the language model and lm_head tensors, renames `model.language_model.*` to `model.*`,
and writes the text config as a standalone `Gemma4ForCausalLM` config (model_type
gemma4_text). The quantization_config is kept with its ignore list reduced to text modules.
Everything else (tokenizer, chat template, generation config) is copied unchanged.

Usage: python3 strip_multimodal.py SRC_DIR DST_DIR
"""

import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

TEXT_PREFIX = "model.language_model."
COPY_FILES = ["chat_template.jinja", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]


# Copy the text tensors under their text-only names; return (kept bytes, dropped bytes).
def strip_tensors(src: str, dst: str) -> tuple[int, int]:
    kept, dropped, tensors = 0, 0, {}
    with safe_open(os.path.join(src, "model.safetensors"), framework="pt") as source:
        for key in source.keys():
            tensor = source.get_tensor(key)
            size = tensor.numel() * tensor.element_size()
            if key.startswith(TEXT_PREFIX):
                tensors["model." + key[len(TEXT_PREFIX):]] = tensor
                kept += size
            elif key.startswith("lm_head."):
                tensors[key] = tensor
                kept += size
            else:
                dropped += size
    save_file(tensors, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    return kept, dropped


# Build the text-only config from the multimodal one.
def text_config(src: str) -> dict:
    with open(os.path.join(src, "config.json"), encoding="utf-8") as handle:
        full = json.load(handle)
    config = dict(full["text_config"])
    config["architectures"] = ["Gemma4ForCausalLM"]
    config["model_type"] = config.get("model_type", "gemma4_text")
    config["tie_word_embeddings"] = full.get("tie_word_embeddings", config.get("tie_word_embeddings"))
    config["eos_token_id"] = full.get("eos_token_id", config.get("eos_token_id"))
    # "vision" bidirectional attention only matters inside image blocks; with the vision tower gone it would only make the
    # exported engine refuse chunked prefill and prefix reuse (Gemma 4 12B).
    config["use_bidirectional_attention"] = None  # explicit: the model class would otherwise default it back to "vision"
    quant = full.get("quantization_config")
    if quant:
        quant = dict(quant)
        quant["ignore"] = [
            name.replace("model.language_model.", "model.") for name in quant.get("ignore", [])
            if name.startswith("model.language_model.") or name == "lm_head"
        ]
        config["quantization_config"] = quant
    return config


# Strip SRC into DST and print what was kept and dropped.
def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    kept, dropped = strip_tensors(src, dst)
    with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(text_config(src), handle, indent=2)
    for name in COPY_FILES:
        if os.path.exists(os.path.join(src, name)):
            shutil.copy(os.path.join(src, name), os.path.join(dst, name))
    print(json.dumps({"kept_gb": round(kept / 1e9, 3), "dropped_gb": round(dropped / 1e9, 3)}))


if __name__ == "__main__":
    main()

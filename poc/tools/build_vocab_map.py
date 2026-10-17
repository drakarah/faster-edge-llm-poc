"""Build a reduced output vocabulary (Edge-LLM `vocab_map.safetensors`) from multilingual text.

The lm_head of Gemma 4 has 262144 rows for ~140 languages; a Dutch/French/English deployment
uses a small fraction. This tool keeps the N most useful output tokens and writes the map that
`tensorrt-edgellm-export --reduced-vocab-dir` consumes (the export slices the lm_head rows).

Selection:
- always kept: every added/special token and the 256 byte tokens (Gemma's byte fallback <0x00>..<0xFF>
  or the byte-level BPE alphabet of Qwen-style tokenizers), so
  any string stays producible (in more tokens when a word's usual pieces are missing);
- the rest by frequency, each language's counts normalised to its own total so a large
  language doesn't crowd out the others (equal weight per text source);
- case-aware: keeping a token also keeps its lower / Title / UPPER variants when they exist
  (without this, "HELLO WORLD" became unproducible at 64k and the model derailed);
- the size is rounded up to a multiple of 128 (Edge-LLM requires it for packed INT4 heads).

Reports coverage per language on held-out articles: the share of tokens (as tokenised with the
full vocabulary) that are inside the map. Higher is better; a token outside the map can only be
produced by spelling it with other tokens.

Usage:
  python3 build_vocab_map.py TOKENIZER_JSON OUT_DIR --size 65536 \
      --text nl=wiki_nl.parquet --text fr=wiki_fr.parquet --text en=wiki_en.parquet \
      [--size 32768 ...] [--extra-text extra.txt] [--articles 20000] [--heldout 1000]
"""

import argparse
import collections
import json
import os

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer


# Read the first `count + heldout` article texts of a Wikipedia parquet shard.
def read_articles(path: str, count: int, heldout: int) -> tuple[list[str], list[str]]:
    table = pq.read_table(path, columns=["text"])
    texts = table.column("text").to_pylist()[: count + heldout]
    return texts[:count], texts[count:]


# Count token ids over texts with the full tokenizer.
def count_tokens(tokenizer: Tokenizer, texts: list[str]) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for start in range(0, len(texts), 512):
        for encoding in tokenizer.encode_batch(texts[start:start + 512], add_special_tokens=False):
            counter.update(encoding.ids)
    return counter


# GPT-2 byte-level BPE alphabet: the printable character standing for each byte (Qwen, Llama 3, ...).
def byte_level_chars() -> list[str]:
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    chars, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            chars[byte] = chr(byte)
        else:
            chars[byte] = chr(256 + extra)
            extra += 1
    return [chars[byte] for byte in range(256)]


# Return the ids of all special/added tokens and the 256 byte tokens: SentencePiece byte fallback (<0x00>..<0xFF>,
# Gemma) or the byte-level BPE alphabet (Qwen), so any string stays producible.
def mandatory_ids(tokenizer: Tokenizer, tokenizer_json: str) -> set[int]:
    with open(tokenizer_json, encoding="utf-8") as handle:
        spec = json.load(handle)
    ids = {token["id"] for token in spec.get("added_tokens", [])}
    vocab = tokenizer.get_vocab()
    for byte, char in enumerate(byte_level_chars()):
        for form in (f"<0x{byte:02X}>", char):
            token_id = vocab.get(form)
            if token_id is not None:
                ids.add(token_id)
    return ids


# Map each token id to the ids of its lower / Title / UPPER case variants that exist.
def build_case_variants(tokenizer: Tokenizer) -> dict[int, list[int]]:
    vocab = tokenizer.get_vocab()
    variants: dict[int, list[int]] = {}
    for token, token_id in vocab.items():
        forms = {token.lower(), token.upper(), token[:1] + token[1:2].upper() + token[2:].lower()
                 if token.startswith(("\u2581", "\u0120")) else token[:1].upper() + token[1:].lower()}
        found = [vocab[form] for form in forms if form != token and form in vocab]
        if found:
            variants[token_id] = found
    return variants


# Build, save and report the reduced vocabulary map.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tokenizer_json")
    parser.add_argument("out_dir")
    parser.add_argument("--size", type=int, action="append", required=True,
                        help="reduced size; repeat for several maps (written to OUT_DIR/v<size>)")
    parser.add_argument("--text", action="append", required=True, help="LANG=parquet path")
    parser.add_argument("--extra-text", action="append", default=[], help="plain-text file, weighted as its own source")
    parser.add_argument("--articles", type=int, default=20000)
    parser.add_argument("--heldout", type=int, default=1000)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    keep = mandatory_ids(tokenizer, args.tokenizer_json)

    scores: collections.Counter = collections.Counter()
    heldout: dict[str, list[str]] = {}
    for item in args.text:
        language, path = item.split("=", 1)
        build, heldout[language] = read_articles(path, args.articles, args.heldout)
        counts = count_tokens(tokenizer, build)
        total = sum(counts.values())
        for token_id, count in counts.items():
            scores[token_id] += count / total
    for path in args.extra_text:
        with open(path, encoding="utf-8") as handle:
            counts = count_tokens(tokenizer, [handle.read()])
        total = sum(counts.values())
        for token_id, count in counts.items():
            scores[token_id] += count / total

    heldout_counts = {}
    for language, texts in heldout.items():
        heldout_counts[language] = count_tokens(tokenizer, texts)
    ranked = [token_id for token_id, _ in scores.most_common()]
    case_variants = build_case_variants(tokenizer)

    for requested in args.size:
        size = -(-max(requested, len(keep)) // 128) * 128
        selected = set(keep)
        for token_id in ranked:
            if len(selected) >= size:
                break
            for variant in [token_id, *case_variants.get(token_id, [])]:
                if len(selected) < size:
                    selected.add(variant)
        vocab_map = torch.tensor(sorted(selected), dtype=torch.int32)

        coverage = {}
        for language, counts in heldout_counts.items():
            total = sum(counts.values())
            coverage[language] = round(100 * sum(c for t, c in counts.items() if t in selected) / total, 3)

        out_dir = os.path.join(args.out_dir, f"v{size}")
        os.makedirs(out_dir, exist_ok=True)
        save_file({"vocab_map": vocab_map}, os.path.join(out_dir, "vocab_map.safetensors"))
        info = {"reduced_vocab_size": len(selected), "original_vocab_size": vocab_size,
                "mandatory_tokens": len(keep), "sources": args.text + args.extra_text,
                "articles_per_language": args.articles, "heldout_coverage_pct": coverage}
        with open(os.path.join(out_dir, "reduced_vocab.json"), "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)
        print(json.dumps(info))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Timing harness for an OpenAI-compatible chat server; every number is measured from the client, the same way for every stack.

  bench.py run    --url URL --label NAME --out FILE [--calls N]    one measurement round against one server
  bench.py report --results DIR --model M --profile P              aggregate the rounds into a table (stock vs optimised)

Workloads (all greedy, temperature 0):
  prefill  a fresh ~1.5k-token prompt (a unique nonce defeats prefix caching); prompt tokens / time to first token
  decode   a long free-text generation; completion tokens after the first / time between first and last token
  replay   the first N of 396 recorded requests (structured JSON answers, tool-style prompts), sent one after another
           in recorded order so prefix reuse sees a realistic sequence; total wall time
"""
import argparse, difflib, glob, json, os, statistics, subprocess, sys, time, urllib.request, uuid

DECODE_PROMPT = ("Write a detailed essay of at least 600 words on how a bicycle gearbox works, from the pedal "
                 "to the wheel. Use continuous prose, no lists.")
FILLER = ("Section {n}. The committee reviews the quarterly maintenance plan, compares the vendor quotes, and "
          "records which items are deferred to the next budget cycle. ")


def post(url, body, stream=False, timeout=3600):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(url, messages, max_tokens):
    """Returns (ttft_s, decode_s, completion_tokens, prompt_tokens)."""
    body = {"model": "local", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = last = None; pt = ct = 0
    with post(url, body) as f:
        for raw in f:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                pt, ct = d["usage"].get("prompt_tokens", 0), d["usage"].get("completion_tokens", 0)
            if d.get("choices") and (d["choices"][0].get("delta") or {}).get("content"):
                now = time.perf_counter()
                first = first or now; last = now
    if first is None:
        raise RuntimeError("no tokens streamed")
    return first - t0, last - first, ct, pt


def prefill(url, runs=3):
    out = []
    for _ in range(runs):
        text = f"[{uuid.uuid4()}] " + "".join(FILLER.format(n=i) for i in range(1, 46)) + "\nSummarise in one word."
        ttft, _, _, pt = stream_chat(url, [{"role": "user", "content": text}], 1)
        out.append(pt / ttft)
    return statistics.median(out), pt


def decode(url, runs=3):
    out = []
    for _ in range(runs):
        _, dt, ct, _ = stream_chat(url, [{"role": "user", "content": f"[{uuid.uuid4()}] " + DECODE_PROMPT}], 400)
        out.append((ct - 1) / dt)
    return statistics.median(out)


def content(resp):
    try:
        m = resp["choices"][0]["message"]
        return (m.get("content") or "") + json.dumps(m.get("tool_calls") or "")
    except (KeyError, IndexError, TypeError):
        return ""


def replay(url, workload, calls):
    recs = [r for r in map(json.loads, open(workload)) if r.get("status") == 200][:calls]
    times, answers, ctoks, ptoks = [], [], 0, 0
    for r in recs:
        body = dict(r["request"]); body["model"] = "local"; body["stream"] = False; body["temperature"] = 0
        t0 = time.perf_counter()
        with post(url, body) as f:
            resp = json.loads(f.read())
        times.append(time.perf_counter() - t0)
        answers.append(content(resp))
        u = resp.get("usage") or {}
        ctoks += u.get("completion_tokens", 0); ptoks += u.get("prompt_tokens", 0)
    return {"seconds": sum(times), "calls": len(times), "completion_tokens": ctoks, "prompt_tokens": ptoks,
            "answers": answers}


def gpu_memory_mib():
    try:
        v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.split()[0]
        return int(v)
    except Exception:
        return None


def cmd_run(a):
    res = {"label": a.label, "url": a.url}
    # warm-up: first request pays one-time costs (graph capture, allocator growth) in every stack
    stream_chat(a.url, [{"role": "user", "content": "Say hello."}], 8)
    res["prefill_tok_s"], res["prefill_prompt_tokens"] = prefill(a.url)
    res["decode_tok_s"] = decode(a.url)
    r = replay(a.url, a.workload, a.calls)
    res.update({"replay_s": r["seconds"], "replay_calls": r["calls"], "replay_completion_tokens": r["completion_tokens"],
                "replay_prompt_tokens": r["prompt_tokens"], "answers": r["answers"], "gpu_memory_mib": gpu_memory_mib()})
    json.dump(res, open(a.out, "w"))
    print(f"{a.label}: prefill {res['prefill_tok_s']:.0f} tok/s, decode {res['decode_tok_s']:.1f} tok/s, "
          f"replay-{r['calls']} {r['seconds']:.1f} s")


def med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(v) if v else None


def cmd_report(a):
    arms = {}
    for f in sorted(glob.glob(os.path.join(a.results, f"{a.model}-{a.profile}-*-round*.json"))):
        arm = os.path.basename(f).split("-")[-2]
        arms.setdefault(arm, []).append(json.load(open(f)))
    if "fast" not in arms:
        sys.exit("no results for the optimised stack")
    stock, fast = arms.get("stock", []), arms["fast"]
    sim = None
    if stock:
        pairs = [(x, y) for x, y in zip(stock[0]["answers"], fast[0]["answers"])]
        same = sum(x == y for x, y in pairs)
        sim = (same, len(pairs), statistics.mean(difflib.SequenceMatcher(None, x, y).ratio() for x, y in pairs))
    rows = [("Prefill, tokens/s (higher is better)", "prefill_tok_s", "{:,.0f}", True),
            ("Decode, tokens/s (higher is better)", "decode_tok_s", "{:.1f}", True),
            (f"Replay of {fast[0]['replay_calls']} recorded requests, seconds (lower is better)", "replay_s", "{:.1f}", False),
            ("GPU memory used, MiB", "gpu_memory_mib", "{:,.0f}", False)]
    out = [f"### {a.model} on {a.profile}: stock Edge-LLM vs optimised fork ({len(fast)} rounds, medians)", "",
           "| | stock | optimised | speed-up |", "|---|---:|---:|---:|"]
    for name, key, fmt, higher in rows:
        s, f_ = med(stock, key), med(fast, key)
        if f_ is None:
            continue
        up = "" if not s else f"**{(f_ / s if higher else s / f_):.2f}x**"
        out.append(f"| {name} | {fmt.format(s) if s else 'did not run'} | {fmt.format(f_)} | {up} |")
    if sim:
        out += ["", f"Greedy answers identical to stock: {sim[0]}/{sim[1]} (mean text similarity {sim[2]:.3f}). "
                "Quantised weights are the same; the optimised stack also uses an INT8 KV cache and fused kernels, so some "
                "answers differ in wording. Read the two answers of the first request in `results/`."]
    text = "\n".join(out)
    print(text)
    open(os.path.join(a.results, f"{a.model}-{a.profile}.md"), "w").write(text + "\n")


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--url", required=True); r.add_argument("--label", required=True)
    r.add_argument("--out", required=True); r.add_argument("--calls", type=int, default=60)
    r.add_argument("--workload", default=os.path.join(os.path.dirname(__file__), "data", "workload.jsonl"))
    q = sub.add_parser("report"); q.add_argument("--results", required=True)
    q.add_argument("--model", required=True); q.add_argument("--profile", required=True)
    a = p.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()

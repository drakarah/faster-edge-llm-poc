#!/usr/bin/env python3
"""Timing harness for an OpenAI-compatible chat server; every number is measured from the client, the same way for every stack.

  bench.py run    --url URL --label NAME --out FILE [--calls N]    one measurement round against one server
  bench.py report --results DIR --model M --profile P              aggregate the rounds into a table (stock vs optimised)

Workloads (all greedy, temperature 0):
  prefill  a fresh ~1.5k-token prompt (a unique nonce defeats prefix caching); prompt tokens / time to first token (cold), and
           the time to first token of the same prompt sent again (warm, prefix reused)
  decode   a long free-text generation; completion tokens after the first / time between first and last token; reported cold (first
           request) and warm (same prompt again) separately
  humaneval  optional (--humaneval N): the public HumanEval benchmark, pass@1 and generation tokens/s (poc/humaneval.py)
  longctx  optional (--longctx 4096,16384,...): prompts of about that many tokens (distinct records, a unique nonce in front): cold time to
           first token and prompt tokens/s, the same prompt again (warm), needle retrieval at 10/50/90% depth (exact match), and decode
           tokens/s with the long context in the KV cache; a stack that refuses or fails a length is recorded as such and skipped for longer ones
  replay   the first N of 396 recorded requests (structured JSON answers, tool-style prompts), sent one after another
           in recorded order so prefix reuse sees a realistic sequence; total wall time
"""
import humaneval
import argparse, difflib, glob, json, os, re, statistics, subprocess, sys, time, urllib.error, urllib.request, uuid

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
    body.update(json.loads(os.environ.get("BENCH_EXTRA_BODY") or "{}"))   # e.g. {"chat_template_kwargs": {"enable_thinking": false}} for servers that think by default
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
    """Cold: a fresh ~1.5k-token prompt (unique nonce, nothing to reuse), prompt tokens / time to first token, median of runs.
    Warm: the identical prompt sent again right after; both stacks run with prefix reuse, so only the new tokens are computed.
    Reported as time to first token (the prompt-token rate would divide by tokens that were never computed)."""
    cold, warm = [], []
    for _ in range(runs):
        text = f"[{uuid.uuid4()}] " + "".join(FILLER.format(n=i) for i in range(1, 46)) + "\nSummarise in one word."
        msgs = [{"role": "user", "content": text}]
        ttft, _, _, pt = stream_chat(url, msgs, 1)
        cold.append(pt / ttft)
        warm.append(stream_chat(url, msgs, 1)[0])
    return statistics.median(cold), pt, statistics.median(warm)


def decode(url, repeats=2):
    """The same essay prompt is sent 1 + repeats times (a fresh nonce each time defeats prefix caching, the text is identical).
    First request = cold: the engine has not seen this text. Repeats = warm: the optimised stack's suffix drafting finds the earlier
    answer's phrases again and accepts longer drafts, so these run faster there. The two are reported separately, never averaged."""
    rates = []
    for _ in range(1 + repeats):
        _, dt, ct, _ = stream_chat(url, [{"role": "user", "content": f"[{uuid.uuid4()}] " + DECODE_PROMPT}], 400)
        rates.append((ct - 1) / dt)
    return rates[0], statistics.median(rates[1:])


def content(resp):
    try:
        m = resp["choices"][0]["message"]
        return (m.get("content") or "") + json.dumps(m.get("tool_calls") or "")
    except (KeyError, IndexError, TypeError):
        return ""


def replay(url, workload, calls):
    recs = [r for r in map(json.loads, open(workload)) if r.get("status") == 200][:calls]
    skip = {int(x) for x in os.environ.get("BENCH_REPLAY_SKIP", "").split(",") if x.strip()}   # request indices to leave out (to time the requests another server accepted)
    recs = [r for i, r in enumerate(recs) if i not in skip]
    times, answers, ctoks, ptoks, errors = [], [], 0, 0, []
    for i, r in enumerate(recs):
        body = dict(r["request"]); body["model"] = "local"; body["stream"] = False; body["temperature"] = 0
        body.update(json.loads(os.environ.get("BENCH_EXTRA_BODY") or "{}"))
        t0 = time.perf_counter()
        try:
            with post(url, body) as f:
                resp = json.loads(f.read())
        except urllib.error.HTTPError as e:  # a stack may refuse a request the other accepts: count it, keep going
            errors.append(f"request {i}: HTTP {e.code} {e.read()[:300].decode(errors='replace')}")
            print(errors[-1], flush=True)
            answers.append("")
            continue
        times.append(time.perf_counter() - t0)
        answers.append(content(resp))
        u = resp.get("usage") or {}
        ctoks += u.get("completion_tokens", 0); ptoks += u.get("prompt_tokens", 0)
    return {"seconds": sum(times), "calls": len(times), "errors": errors, "completion_tokens": ctoks, "prompt_tokens": ptoks,
            "answers": answers}


RECORD = "Record {i}: the valve in bay {i} is set to {v} bar. "


def record_value(i):
    return (i * 37) % 900 + 100


def long_prompt(n, nonce, question):
    body = "".join(RECORD.format(i=i, v=record_value(i)) for i in range(1, n + 1))
    return f"[{nonce}] Read the following maintenance records.\n{body}\n\n{question}"


def longctx(url, lengths):
    """One entry per target length: cold TTFT, prompt tokens, warm TTFT, needles correct of 3, decode tok/s. The record count is
    calibrated against the server's own prompt token count so every stack sees (about) the same number of tokens."""
    out = {}
    per = 17.0  # tokens per record, refined after the first request
    for target in lengths:
        entry = {}
        try:
            n = max(8, int(target / per))
            q = f"What is the valve in bay {n // 2} set to? Answer with the number only."
            for _ in range(2):  # calibration, and an untimed second pass: a freshly started server pays one-time costs on its first long prompts (28 s on the first 1k-token one)
                _, _, _, pt = stream_chat(url, [{"role": "user", "content": long_prompt(n, uuid.uuid4(), q)}], 1)
            per = pt / n
            n = max(8, int(target / per))
            colds, tokens, ok, first_msgs = [], 0, 0, None
            for depth in (0.1, 0.5, 0.9):
                k = max(1, int(n * depth))
                msgs = [{"role": "user", "content": long_prompt(n, uuid.uuid4(), f"What is the valve in bay {k} set to? Answer with the number only.")}]
                first_msgs = first_msgs or msgs
                body = {"model": "local", "messages": msgs, "max_tokens": 12, "temperature": 0}
                body.update(json.loads(os.environ.get("BENCH_EXTRA_BODY") or "{}"))
                t0 = time.perf_counter()
                with post(url, body) as f:
                    resp = json.loads(f.read())
                colds.append(time.perf_counter() - t0)
                tokens = (resp.get("usage") or {}).get("prompt_tokens", 0)
                ok += str(record_value(k)) in re.findall(r"\d+", content(resp))[:1]
            entry.update({"prompt_tokens": tokens, "cold_s": statistics.median(colds), "cold_tok_s": tokens / statistics.median(colds), "needles_ok": ok,
                          "warm_ttft_s": stream_chat(url, first_msgs, 1)[0]})
            essay = [{"role": "user", "content": long_prompt(n, uuid.uuid4(), "Now write an essay of at least 300 words on how a pressure valve works. Use continuous prose, no lists.")}]
            _, dt, ct, _ = stream_chat(url, essay, 300)
            entry["decode_tok_s"] = (ct - 1) / dt
        except Exception as e:  # refused (HTTP 4xx/5xx), out of memory, server gone: record it and do not try longer prompts
            body = e.read()[:200].decode(errors="replace") if isinstance(e, urllib.error.HTTPError) else ""
            entry["error"] = f"{type(e).__name__} {e} {body}".strip()[:300]
            out[str(target)] = entry
            print(f"longctx {target}: {entry['error']}", flush=True)
            break
        out[str(target)] = entry
        print(f"longctx {target}: {entry}", flush=True)
    return out


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
    res["prefill_tok_s"], res["prefill_prompt_tokens"], res["prefill_warm_ttft_s"] = prefill(a.url)
    res["decode_cold_tok_s"], res["decode_warm_tok_s"] = decode(a.url)
    r = replay(a.url, a.workload, a.calls)
    res.update({"replay_s": r["seconds"], "replay_calls": r["calls"], "replay_completion_tokens": r["completion_tokens"],
                "replay_prompt_tokens": r["prompt_tokens"], "replay_errors": r["errors"], "answers": r["answers"], "gpu_memory_mib": gpu_memory_mib()})
    if a.humaneval:
        res.update(humaneval.run_humaneval(a.url, os.path.join(os.path.dirname(a.out), "humaneval-problems.json"), a.humaneval))
        print(f"{a.label}: HumanEval pass@1 {res['he_pass']}/{res['he_total']}, {res['he_tok_s']:.1f} tok/s")
    json.dump(res, open(a.out, "w"))
    if a.longctx:
        res["longctx"] = longctx(a.url, [int(x) for x in a.longctx.split(",") if x])
        json.dump(res, open(a.out, "w"))
    print(f"{a.label}: prefill {res['prefill_tok_s']:.0f} tok/s cold / warm TTFT {res['prefill_warm_ttft_s']*1000:.0f} ms, decode {res['decode_cold_tok_s']:.1f} cold / {res['decode_warm_tok_s']:.1f} warm tok/s, "
          f"replay-{r['calls']} {r['seconds']:.1f} s" + (f", {len(r['errors'])} requests refused" if r["errors"] else ""))


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
    he = fast[0].get("he_total")
    rows = [("Prefill of a fresh ~1.5k-token prompt (cold), tokens/s (higher is better)", "prefill_tok_s", "{:,.0f}", True),
            ("Same prompt again (warm, prefix reused), time to first token, seconds (lower is better)", "prefill_warm_ttft_s", "{:.3f}", False),
            ("Decode, first request of a prompt (cold), tokens/s (higher is better)", "decode_cold_tok_s", "{:.1f}", True),
            ("Decode, same prompt again (warm), tokens/s (higher is better)", "decode_warm_tok_s", "{:.1f}", True),
            (f"Replay of {fast[0]['replay_calls']} recorded requests, seconds (lower is better)", "replay_s", "{:.1f}", False),
            ("Replay output tokens in total (the stacks do not answer with the same length)", "replay_completion_tokens", "{:,.0f}", False),
            *([("HumanEval generation, tokens/s (higher is better)", "he_tok_s", "{:.1f}", True)] if he else []),
            ("GPU memory used, MiB", "gpu_memory_mib", "{:,.0f}", False)]
    if stock:
        out = [f"### {a.model} on {a.profile}: stock Edge-LLM vs optimised fork ({len(fast)} rounds, medians)", "",
               "| | stock | optimised | speed-up |", "|---|---:|---:|---:|"]
    else:   # no stock arm (the model does not run on stock Edge-LLM): optimised column only
        out = [f"### {a.model} on {a.profile}: optimised fork, no stock arm ({len(fast)} rounds, medians)", "",
               "| | optimised |", "|---|---:|"]
    for name, key, fmt, higher in rows:
        s, f_ = med(stock, key), med(fast, key)
        if f_ is None:
            continue
        up = "" if (not s or key == "replay_completion_tokens") else f"**{(f_ / s if higher else s / f_):.2f}x**"
        out.append(f"| {name} | {fmt.format(s) if s else 'did not run'} | {fmt.format(f_)} | {up} |" if stock else f"| {name} | {fmt.format(f_)} |")
    if he:
        def hp(rs):
            return "did not run" if not rs else " / ".join(f"{r['he_pass']}/{r['he_total']}" for r in rs)
        out += ["", f"HumanEval pass@1 (greedy, public benchmark), per round: " + (f"stock {hp(stock)}; " if stock else "") + f"optimised {hp(fast)}."]
    refused = {n: sum(len(r.get("replay_errors", [])) for r in rs) for n, rs in (("stock", stock), ("optimised", fast)) if rs}
    if any(refused.values()):
        out += ["", "Requests refused by a server (not counted in its replay seconds; see `replay_errors` in `results/`): " +
                ", ".join(f"{n} {c}" for n, c in refused.items() if c) + "."]
    if sim:
        out += ["", f"Greedy answers identical to stock: {sim[0]}/{sim[1]} (mean text similarity {sim[2]:.3f}). "
                "Quantised weights are the same; the optimised stack also uses an INT8 KV cache and fused kernels, so some "
                "answers differ in wording. Read the two answers of the first request in `results/`."]
    lc = sorted({int(k) for rs in (stock, fast) for r in rs for k in r.get("longctx", {})})
    if lc:
        def cell(rs, L, key, fmt):
            vals = [r["longctx"][str(L)] for r in rs if str(L) in r.get("longctx", {})]
            good = [v[key] for v in vals if key in v]
            if good:
                return fmt.format(statistics.median(good))
            return "refused / failed" if vals else "not reached"
        out += ["", "#### Long context (prompts of the given length, greedy; cold = nothing to reuse, warm = same prompt again, needles = exact retrieval of 3 planted facts)", "",
                *(["| prompt tokens | metric | stock | optimised |", "|---:|---|---:|---:|"] if stock else ["| prompt tokens | metric | optimised |", "|---:|---|---:|"])]
        for L in lc:
            ptoks = med([r.get("longctx", {}).get(str(L), {}) for r in fast], "prompt_tokens")
            for name, key, fmt in (("cold time to first token, s", "cold_s", "{:.2f}"), ("cold prompt tokens/s", "cold_tok_s", "{:,.0f}"),
                                   ("warm time to first token, s", "warm_ttft_s", "{:.2f}"), ("decode tokens/s at this context", "decode_tok_s", "{:.1f}"),
                                   ("needles correct of 3", "needles_ok", "{:.0f}")):
                label = f"{L:,}" + (f" (actual {ptoks:,.0f})" if ptoks and key == "cold_s" else "")
                out.append(f"| {label if key == 'cold_s' else ''} | {name} | {cell(stock, L, key, fmt)} | {cell(fast, L, key, fmt)} |" if stock
                           else f"| {label if key == 'cold_s' else ''} | {name} | {cell(fast, L, key, fmt)} |")
        errs = sorted({f"{n} {L}: {v['error']}" for n, rs in (("stock", stock), ("optimised", fast)) for r in rs for L, v in r.get("longctx", {}).items() if "error" in v})
        if errs:
            out += ["", "Long-context refusals / failures: " + "; ".join(errs[:6])]
    text = "\n".join(out)
    print(text)
    open(os.path.join(a.results, f"{a.model}-{a.profile}.md"), "w").write(text + "\n")


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--url", required=True); r.add_argument("--label", required=True)
    r.add_argument("--out", required=True); r.add_argument("--calls", type=int, default=60)
    r.add_argument("--humaneval", type=int, default=0, help="HumanEval problems to run (0 = skip, max 164)")
    r.add_argument("--longctx", default="", help="comma-separated target prompt lengths in tokens for the long-context test (empty = skip)")
    r.add_argument("--workload", default=os.path.join(os.path.dirname(__file__), "data", "workload.jsonl"))
    q = sub.add_parser("report"); q.add_argument("--results", required=True)
    q.add_argument("--model", required=True); q.add_argument("--profile", required=True)
    a = p.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()

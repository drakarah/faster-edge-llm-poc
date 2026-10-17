#!/usr/bin/env python3
"""HumanEval pass@1 (greedy) against an OpenAI-compatible chat server: public benchmark, objective pass/fail.

The 164 problems come from the public Hugging Face dataset openai/openai_humaneval (fetched once through the dataset
server API, cached next to the results). The model gets the signature and docstring and must answer with the complete
function in one python code block. Each answer is run together with the problem's unit tests in a child python process
(empty environment, 20 s and 2 GiB limits, scratch directory), which is the standard way to score it. Model-written code
runs inside this container, so run the PoC on a machine you are happy to run generated python on.
"""
import json, os, re, resource, subprocess, sys, tempfile, time, urllib.request

ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset=openai/openai_humaneval"
            "&config=openai_humaneval&split=test&offset={offset}&length=100")
PROMPT = ("Complete the following Python function. Reply with the complete function (imports and signature included) "
          "in a single ```python code block and nothing else.\n\n```python\n{prompt}\n```")


def load_tasks(cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    tasks = []
    for offset in (0, 100):
        with urllib.request.urlopen(ROWS_URL.format(offset=offset), timeout=120) as f:
            tasks += [r["row"] for r in json.load(f)["rows"]]
    json.dump(tasks, open(cache, "w"))
    return tasks


def ask(url, prompt, max_tokens):
    body = {"model": "local", "messages": [{"role": "user", "content": PROMPT.format(prompt=prompt)}],
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as f:
        reply = json.load(f)
    return reply["choices"][0]["message"]["content"] or "", reply["usage"]["completion_tokens"]


def extract(text, task):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    code = blocks[0] if blocks else text
    if f"def {task['entry_point']}" not in code:
        code = task["prompt"] + code
    return code


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))


def passes(code, task, timeout=20):
    program = code + "\n\n" + task["test"] + f"\n\ncheck({task['entry_point']})\n"
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "t.py")
        open(path, "w").write(program)
        try:
            r = subprocess.run([sys.executable, path], cwd=scratch, env={}, capture_output=True,
                               timeout=timeout, preexec_fn=_limits)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def run_humaneval(url, cache, limit=164, max_tokens=1024):
    tasks = load_tasks(cache)[:limit]
    outputs, t0 = [], time.perf_counter()
    for task in tasks:
        try:
            text, tokens = ask(url, task["prompt"], max_tokens)
        except Exception as error:  # a server error counts as a failed task
            print(task["task_id"], "request failed:", error, flush=True)
            text, tokens = "", 0
        outputs.append((task, text, tokens))
    seconds = time.perf_counter() - t0
    ok = [passes(extract(text, task), task) for task, text, _ in outputs]
    tokens = sum(t for _, _, t in outputs)
    return {"he_pass": sum(ok), "he_total": len(tasks), "he_tokens": tokens, "he_seconds": seconds,
            "he_tok_s": tokens / seconds if seconds else None,
            "he_results": [{"task_id": t["task_id"], "pass": o, "completion": x} for (t, x, _), o in zip(outputs, ok)]}

#!/usr/bin/env python
"""Evaluate a model on the ShaderForge test splits.

    python eval/evaluate.py --reference --split test_iid --limit 50
    python eval/evaluate.py --model path/or/hf-id --label grpo
    python eval/evaluate.py --endpoint https://openrouter.ai/api/v1 --api-model <id> --label frontier
        (API key is read from env OPENROUTER_API_KEY, or the variable named by --key-env)

Metrics per split (failures count as similarity 0):
    compile_rate  fraction of completions that compiled and rendered
    mean_sim      mean similarity (0 for failures)
    pass@1        fraction with similarity >= 0.85
    mean_reward   mean of the training reward
with bootstrap 95% CIs. Rows accumulate in eval/results.json and eval/results.md.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shaderforge import data as D  # noqa: E402
from shaderforge.reward import SIM_PASS, as_block, score_completions  # noqa: E402
from shaderforge.render import RenderPool  # noqa: E402

TEST_SPLITS = ["test_iid", "test_unseen_params", "test_unseen_combo"]


def bootstrap_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(results):
    compiled = [r.stage in ("scored", "blank") for r in results]
    sim = [r.similarity if r.stage == "scored" else 0.0 for r in results]
    passed = [float(s >= SIM_PASS) for s in sim]
    reward = [r.reward for r in results]
    out = {"n": len(results)}
    for name, vals in [("compile_rate", [float(c) for c in compiled]), ("mean_sim", sim),
                       ("pass@1", passed), ("mean_reward", reward)]:
        m, lo, hi = bootstrap_ci(vals)
        out[name] = {"mean": m, "lo": lo, "hi": hi}
    return out


# ---------------------------------------------------------------- generators
def gen_reference(tasks):
    return [as_block(t["reference"]) for t in tasks]


def gen_vllm(tasks, model_path, max_new_tokens, temperature, gpu_mem=0.85):
    """Fast path: all prompts in one vLLM call. Returns None if vLLM is unavailable."""
    try:
        from vllm import LLM, SamplingParams
    except Exception:  # noqa: BLE001
        return None
    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_mem,
              max_model_len=2048, seed=0, enforce_eager=False)
    sp = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, seed=0)
    msgs = [[{"role": "user", "content": t["prompt"]}] for t in tasks]
    outs = llm.chat(msgs, sp, use_tqdm=False)
    res = [o.outputs[0].text for o in outs]
    del llm
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return res


def gen_hf(tasks, model_path, max_new_tokens, batch_size, temperature):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    outs = []
    for i in range(0, len(tasks), batch_size):
        chunk = tasks[i:i + batch_size]
        texts = [tok.apply_chat_template([{"role": "user", "content": t["prompt"]}], tokenize=False,
                                         add_generation_prompt=True) if tok.chat_template else t["prompt"]
                 for t in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(dev)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        kw.update(dict(do_sample=True, temperature=temperature) if temperature > 0 else dict(do_sample=False))
        with torch.no_grad():
            gen = model.generate(**enc, **kw)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
        print(f"  generated {len(outs)}/{len(tasks)}", file=sys.stderr)
    return outs


def _chat_once(endpoint, key, model, prompt, max_tokens, temperature, retries=4):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  request failed: {type(e).__name__}", file=sys.stderr)
                return ""
            time.sleep(2 ** attempt)


def gen_api(tasks, endpoint, model, key_env, max_tokens, temperature, concurrency):
    key = os.environ.get(key_env)
    if not key:
        sys.exit(f"set {key_env} in the environment to use --endpoint")
    with ThreadPoolExecutor(concurrency) as ex:
        return list(ex.map(lambda t: _chat_once(endpoint, key, model, t["prompt"], max_tokens, temperature),
                           tasks))


# ---------------------------------------------------------------- table
def fmt(m, pct=False):
    s = 100 if pct else 1
    d = 1 if pct else 3
    return f"{m['mean'] * s:.{d}f} [{m['lo'] * s:.{d}f}, {m['hi'] * s:.{d}f}]"


def write_tables(rows, md_path):
    lines = ["# Results", "",
             "Mean over tasks, with bootstrap 95% CI in brackets. Failures (no code, compile error, timeout) "
             "count as similarity 0. pass@1 = similarity >= 0.85.", "",
             "| model | split | n | compile % | mean sim | pass@1 % | mean reward |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        s = r["summary"]
        lines.append(f"| {r['label']} | {r['split']} | {s['n']} | {fmt(s['compile_rate'], True)} | "
                     f"{fmt(s['mean_sim'])} | {fmt(s['pass@1'], True)} | {fmt(s['mean_reward'])} |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reference", action="store_true", help="score the reference shaders (upper bound)")
    g.add_argument("--model", help="HF model path / id (local generation)")
    g.add_argument("--endpoint", help="OpenAI-compatible base URL")
    ap.add_argument("--api-model", help="model id for --endpoint")
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--label", default=None)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--split", default="all", help="a split name or 'all' (the three test splits)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "eval"))
    a = ap.parse_args()

    if a.endpoint and not a.api_model:
        ap.error("--endpoint needs --api-model")
    label = a.label or ("reference" if a.reference else (a.api_model or os.path.basename(a.model.rstrip("/"))))
    splits = TEST_SPLITS if a.split == "all" else [a.split]
    pool = RenderPool(a.workers)
    new_rows = []
    split_tasks = {}
    for split in splits:
        path = os.path.join(a.data, f"{split}.jsonl")
        if not os.path.exists(path):
            sys.exit(f"{path} missing: run `python scripts/make_dataset.py` first")
        split_tasks[split] = D.read_jsonl(path)[: a.limit]
    pre = {}
    if a.model and not a.no_vllm:
        flat = [t for s in splits for t in split_tasks[s]]
        comps_all = gen_vllm(flat, a.model, a.max_new_tokens, a.temperature)
        if comps_all is not None:
            k = 0
            for s in splits:
                pre[s] = comps_all[k:k + len(split_tasks[s])]
                k += len(split_tasks[s])
            print(f"[{label}] vLLM generated {len(flat)} completions", file=sys.stderr)
    for split in splits:
        tasks = split_tasks[split]
        print(f"[{label}] {split}: {len(tasks)} tasks", file=sys.stderr)
        if a.reference:
            comps = gen_reference(tasks)
        elif split in pre:
            comps = pre[split]
        elif a.model:
            comps = gen_hf(tasks, a.model, a.max_new_tokens, a.batch_size, a.temperature)
        else:
            comps = gen_api(tasks, a.endpoint, a.api_model, a.key_env, a.max_new_tokens, a.temperature,
                            a.concurrency)
        results = score_completions(comps, [os.path.join(a.data, t["target"]) for t in tasks], pool)
        summ = summarize(results)
        new_rows.append({"label": label, "split": split, "summary": summ})
        stages = {}
        for r in results:
            stages[r.stage] = stages.get(r.stage, 0) + 1
        print(f"{split}: compile {fmt(summ['compile_rate'], True)}  sim {fmt(summ['mean_sim'])}  "
              f"pass@1 {fmt(summ['pass@1'], True)}  stages {stages}")
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, f"samples_{label}_{split}.jsonl"), "w") as f:
            for t, c, r in zip(tasks, comps, results):
                f.write(json.dumps({"id": t["id"], "completion": c, **r.as_dict()}) + "\n")
    pool.close()

    # accumulate rows (replace same label+split), regenerate markdown
    jpath = os.path.join(a.out_dir, "results.json")
    rows = json.load(open(jpath)) if os.path.exists(jpath) else []
    keys = {(r["label"], r["split"]) for r in new_rows}
    rows = [r for r in rows if (r["label"], r["split"]) not in keys] + new_rows
    json.dump(rows, open(jpath, "w"), indent=1)
    write_tables(rows, os.path.join(a.out_dir, "results.md"))
    print("wrote", os.path.join(a.out_dir, "results.md"))


if __name__ == "__main__":
    main()

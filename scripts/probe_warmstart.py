#!/usr/bin/env python
"""Measure held-out compile rate, pass@K, and reward std before starting GRPO."""
import argparse, json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from shaderforge import data as D
from shaderforge.render import RenderPool
from shaderforge.reward import SIM_PASS, score_completions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="ckpt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="train_rl")
    ap.add_argument("--per-family", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.15)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    rows = D.read_jsonl(os.path.join(a.data, f"{a.split}.jsonl"))
    picked = []
    for fam in ("wobble", "metaball", "spiral"):
        picked.extend([r for r in rows if r["family"] == fam][: a.per_family])

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    pool = RenderPool(a.workers)

    summary = {"label": a.label, "model": a.model, "families": {}}
    print(f"=== PROBE {a.label} ({a.model}) k={a.k} T={a.temp} ===", flush=True)
    for fam in ("wobble", "metaball", "spiral"):
        frows = [r for r in picked if r["family"] == fam]
        comp_tot = pass_prompts = total_samples = 0
        stds, best_sims = [], []
        for r in frows:
            ptxt = (tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                            tokenize=False, add_generation_prompt=True)
                    if tok.chat_template else r["prompt"])
            enc = tok([ptxt] * a.k, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=True,
                                     temperature=a.temp, top_p=0.95, pad_token_id=tok.pad_token_id)
            texts = [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen]
            res = score_completions(texts, [os.path.join(a.data, r["target"])] * a.k, pool)
            revs = [x.reward for x in res]
            sims = [x.similarity for x in res]
            comp_tot += sum(x.stage in ("scored", "blank") for x in res)
            total_samples += a.k
            stds.append(float(np.std(revs)))
            best_sims.append(max(sims))
            if max(sims) >= SIM_PASS:
                pass_prompts += 1
        st = {
            "compile_rate": round(comp_tot / max(1, total_samples), 3),
            "pass_at_k_prompts": f"{pass_prompts}/{len(frows)}",
            "mean_group_std": round(float(np.mean(stds)), 4),
            "nonzero_std_prompts": sum(s > 1e-4 for s in stds),
            "mean_best_sim": round(float(np.mean(best_sims)), 3),
        }
        summary["families"][fam] = st
        print(f"  {fam:8s}: compile={st['compile_rate']*100:5.1f}%  pass@{a.k}={st['pass_at_k_prompts']}  "
              f"grp_std={st['mean_group_std']:.3f} ({st['nonzero_std_prompts']}/{len(frows)} >0)  "
              f"best_sim={st['mean_best_sim']:.3f}", flush=True)
    pool.close()
    print("PROBE_JSON:" + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

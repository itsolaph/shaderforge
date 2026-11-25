#!/usr/bin/env python
"""Held-out-family experiment on top of an existing dataset (run make_dataset.py first).

Families in HOLDOUT never get a reference shader in supervised training:
  train_sft_a.jsonl      train rows whose base family is NOT held out (SFT sees only these)
  train_rl.jsonl         GRPO prompts: every held-out-family train row + an equal number of the others
                         (GRPO only needs the target image, never the reference code)
  test_new_family.jsonl  fresh tasks from the held-out families, ids disjoint from every other split
  test_iid_a.jsonl       test_iid restricted to the seen families (regression check)

    python scripts/make_family_split.py --data data --n 150
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shaderforge import data as D  # noqa: E402
from shaderforge import families as F  # noqa: E402
from shaderforge.render import RenderPool, save_png  # noqa: E402
from shaderforge.reward import is_blank  # noqa: E402

HOLDOUT = ("spiral", "metaball", "wobble")


def new_family_tasks(n, seed, used):
    rng = D.split_rng(seed, "test_new_family")
    out, seen = [], set(used)
    while len(out) < n:
        fam = rng.choice(HOLDOUT)
        mod = None
        if rng.random() < 0.4:
            mods = sorted(m for (b, m) in F.SEEN_COMBOS if b == fam)
            mod = rng.choice(mods) if mods else None
        s = {"family": fam, "params": F.FAMILIES[fam].sample(rng, "train"), "modifier": mod,
             "mod_params": F.MODIFIERS[mod].sample(rng, "train") if mod else None}
        t = D.build_task(s, "test_new_family")
        if t.id in seen:
            continue
        seen.add(t.id)
        out.append(t)
    return out


def write_rows(rows, path):
    import json
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    split_rows = {s: D.read_jsonl(os.path.join(a.data, f"{s}.jsonl")) for s in D.SPLITS}
    used = {r["id"] for rows in split_rows.values() for r in rows}
    train = split_rows["train"]
    held = [r for r in train if r["family"] in HOLDOUT]
    seen = [r for r in train if r["family"] not in HOLDOUT]
    rng = random.Random(a.seed)
    rl = held + rng.sample(seen, min(len(seen), len(held)))
    rng.shuffle(rl)
    write_rows(seen, os.path.join(a.data, "train_sft_a.jsonl"))
    write_rows(rl, os.path.join(a.data, "train_rl.jsonl"))
    write_rows([r for r in split_rows["test_iid"] if r["family"] not in HOLDOUT],
               os.path.join(a.data, "test_iid_a.jsonl"))
    print(f"train_sft_a {len(seen)}  train_rl {len(rl)} ({len(held)} held-out-family rows)")

    tasks = new_family_tasks(a.n, a.seed, used)
    pool = RenderPool(a.workers)
    for t, res in zip(tasks, pool.render_many([t.reference for t in tasks], 128)):
        if not res.ok:
            sys.exit(f"reference failed: {t.family} {t.id} {res.stage} {res.error}")
        if is_blank(res.image):
            print(f"warning: blank reference {t.family}/{t.modifier} {t.id}", file=sys.stderr)
        t.target = os.path.join("targets", "test_new_family", f"{t.id}.png")
        save_png(res.image, os.path.join(a.data, t.target))
    pool.close()
    D.write_jsonl(tasks, os.path.join(a.data, "test_new_family.jsonl"))
    fams = {f: sum(t.family == f for t in tasks) for f in HOLDOUT}
    print(f"test_new_family {len(tasks)} {fams}")


if __name__ == "__main__":
    main()

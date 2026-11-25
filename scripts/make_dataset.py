#!/usr/bin/env python
"""Generate the ShaderForge dataset: JSONL splits + 128px target PNGs.

    python scripts/make_dataset.py --small          # tiny smoke set
    python scripts/make_dataset.py                  # 4000 / 200 / 300 / 200 / 200
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shaderforge import data as D  # noqa: E402
from shaderforge.reward import is_blank  # noqa: E402
from shaderforge.render import RenderPool, save_png  # noqa: E402

FULL = {"train": 4000, "val": 200, "test_iid": 300, "test_unseen_params": 200, "test_unseen_combo": 200}
SMALL = {"train": 64, "val": 12, "test_iid": 60, "test_unseen_params": 20, "test_unseen_combo": 20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-render", action="store_true", help="only write JSONL (no target PNGs)")
    a = ap.parse_args()

    counts = SMALL if a.small else FULL
    splits = D.generate_all(counts, a.seed)
    pool = None if a.no_render else RenderPool(a.workers)
    for name, tasks in splits.items():
        if pool:
            results = pool.render_many([t.reference for t in tasks], a.size)
            for t, res in zip(tasks, results):
                if not res.ok:
                    sys.exit(f"reference failed to render ({t.family}/{t.modifier} {t.id}): {res.stage} {res.error}")
                if is_blank(res.image):
                    print(f"warning: blank reference {t.family}/{t.modifier} {t.id}", file=sys.stderr)
                rel = os.path.join("targets", name, f"{t.id}.png")
                save_png(res.image, os.path.join(a.out, rel))
                t.target = rel
        D.write_jsonl(tasks, os.path.join(a.out, f"{name}.jsonl"))
        print(f"{name}: {len(tasks)} tasks")
    if pool:
        print("renderer:", pool.info)
        pool.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build web/gallery.json + web/gallery.js + web/targets/*.png.

Placeholder mode (default): "trained" = the reference shader, "base" = a deliberately degraded
shader (same family, re-sampled parameters, modifier dropped). The page shows a PLACEHOLDER banner.

Real mode: pass --base-samples / --trained-samples (eval/samples_<label>_<split>.jsonl from
evaluate.py); the shaders are the models' own completions and the banner is hidden.
"""
import argparse
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shaderforge import data as D  # noqa: E402
from shaderforge import families as F  # noqa: E402
from shaderforge.reward import as_block, extract_code, score_completions  # noqa: E402
from shaderforge.render import RenderPool, save_png  # noqa: E402

NOOP = "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n    fragColor = vec4(0.5, 0.5, 0.5, 1.0);\n}\n"


def degrade(task: dict) -> str:
    """Plausible 'weak model' shader: right family, wrong parameters, modifier dropped."""
    h = int.from_bytes(hashlib.sha256(task["id"].encode()).digest()[:8], "big")
    rng = random.Random(h)
    fam = F.FAMILIES[task["family"]]
    for _ in range(50):
        params = fam.sample(rng, "train")
        if params != task["params"]:
            break
    return F.build_shader(task["family"], params)


def load_samples(path):
    if not path:
        return None
    return {json.loads(l)["id"]: json.loads(l) for l in open(path) if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--split", default="test_iid")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(ROOT, "web"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--base-samples")
    ap.add_argument("--trained-samples")
    ap.add_argument("--cost", default="$X", help="cost receipt text shown in the header")
    ap.add_argument("--per-family", type=int, default=0,
                    help="take the first N tasks of every family in file order (no cherry-picking)")
    ap.add_argument("--labels", default="base model,trained (GRPO)", help="column titles: before,after")
    a = ap.parse_args()

    placeholder = not (a.base_samples and a.trained_samples)
    tasks = D.read_jsonl(os.path.join(a.data, f"{a.split}.jsonl"))
    # spread over families: first task of each family, then fill
    picked, seen = [], set()
    for t in tasks:
        if t["family"] not in seen:
            seen.add(t["family"])
            picked.append(t)
    picked += [t for t in tasks if t not in picked]
    picked = picked[: a.n]
    if a.per_family:
        fams = sorted({t["family"] for t in tasks})
        picked = [t for f in fams for t in [x for x in tasks if x["family"] == f][: a.per_family]]

    base_s, trained_s = load_samples(a.base_samples), load_samples(a.trained_samples)
    pool = RenderPool(2)
    os.makedirs(os.path.join(a.out, "targets"), exist_ok=True)
    items = []
    for t in picked:
        res = pool.render(t["reference"], a.size)
        if not res.ok:
            sys.exit(f"reference failed: {res.error}")
        save_png(res.image, os.path.join(a.out, "targets", f"{t['id']}.png"))
        target128 = os.path.join(a.data, t["target"])
        if placeholder:
            comps = {"base": as_block(degrade(t)), "trained": as_block(t["reference"])}
        else:
            comps = {"base": base_s.get(t["id"], {}).get("completion", ""),
                     "trained": trained_s.get(t["id"], {}).get("completion", "")}
        entry = {"id": t["id"], "family": t["family"], "modifier": t["modifier"], "spec": t["spec"],
                 "target": f"targets/{t['id']}.png"}
        for k, comp in comps.items():
            r = score_completions([comp], [target128], pool)[0]
            entry[k] = {"code": extract_code(comp) or "", "reward": round(r.reward, 3),
                        "similarity": round(r.similarity, 3), "stage": r.stage}
        items.append(entry)
    pool.close()

    before, after = a.labels.split(",", 1)
    doc = {"placeholder": placeholder, "split": a.split, "cost": a.cost, "labels": [before, after], "items": items}
    with open(os.path.join(a.out, "gallery.json"), "w") as f:
        json.dump(doc, f, indent=1)
    with open(os.path.join(a.out, "gallery.js"), "w") as f:  # file:// friendly (fetch of local json is blocked)
        f.write("window.GALLERY = " + json.dumps(doc) + ";\n")
    print(f"wrote {len(items)} items to {a.out} (placeholder={placeholder})")


if __name__ == "__main__":
    main()

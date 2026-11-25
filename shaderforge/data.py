"""Deterministic dataset generation (train / val / test_iid / test_unseen_params / test_unseen_combo)."""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from . import families as F

SPLITS = ["train", "val", "test_iid", "test_unseen_params", "test_unseen_combo"]

PROMPT_TEMPLATE = """Write a GLSL fragment shader in Shadertoy format that renders the image described below.

Rules:
- Define exactly `void mainImage(out vec4 fragColor, in vec2 fragCoord)`; you may add helper functions above it.
- Use only `iResolution`. No textures, no iChannel, no iTime (the image is static). Do not write #version or a main().
- The image is square. uv = fragCoord / iResolution.xy is in [0,1]^2 with the origin at the bottom-left. p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y is centered on the image, in units of image height, +x right, +y up.
- Colors are given as #rrggbb hex: divide the 0-255 channel values by 255.
- Reply with a single ```glsl code block and nothing else.

Image description:
{spec}"""


def make_prompt(spec: str) -> str:
    return PROMPT_TEMPLATE.format(spec=spec)


@dataclass
class Task:
    id: str
    split: str
    family: str
    modifier: Optional[str]
    params: dict
    mod_params: Optional[dict]
    spec: str
    prompt: str
    reference: str
    target: str = ""          # png path (relative to dataset dir), filled by make_dataset

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def canonical_key(family: str, params: dict, modifier: Optional[str], mod_params: Optional[dict]) -> str:
    blob = json.dumps([family, params, modifier, mod_params], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def split_rng(seed: int, split: str) -> random.Random:
    h = hashlib.sha256(f"{seed}:{split}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def build_task(sampled: dict, split: str) -> Task:
    fam, params, mod, mp = sampled["family"], sampled["params"], sampled["modifier"], sampled["mod_params"]
    spec = F.build_spec(fam, params, mod, mp)
    return Task(
        id=canonical_key(fam, params, mod, mp), split=split, family=fam, modifier=mod,
        params=params, mod_params=mp, spec=spec, prompt=make_prompt(spec),
        reference=F.build_shader(fam, params, mod, mp))


SPLIT_SPEC = {  # regime, mode
    "train": ("train", "mixed"),
    "val": ("train", "mixed"),
    "test_iid": ("train", "mixed"),
    "test_unseen_params": ("unseen", "mixed"),
    "test_unseen_combo": ("train", "held"),
}


def generate_split(split: str, n: int, seed: int = 0, exclude: Optional[set] = None) -> List[Task]:
    """Generate n unique tasks for a split; ids never collide with ``exclude`` (earlier splits)."""
    regime, mode = SPLIT_SPEC[split]
    rng = split_rng(seed, split)
    seen = set(exclude or ())
    out: List[Task] = []
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > n * 200 + 1000:
            raise RuntimeError(f"cannot generate {n} unique tasks for {split}")
        t = build_task(F.sample_task_params(rng, regime, mode), split)
        if t.id in seen:
            continue
        seen.add(t.id)
        out.append(t)
    return out


def generate_all(counts: Dict[str, int], seed: int = 0) -> Dict[str, List[Task]]:
    """Splits are generated in a fixed order so that ids are globally disjoint."""
    used: set = set()
    res: Dict[str, List[Task]] = {}
    for split in SPLITS:
        res[split] = generate_split(split, counts[split], seed, used)
        used |= {t.id for t in res[split]}
    return res


def combos_of(tasks: List[Task]) -> set:
    return {(t.family, t.modifier) for t in tasks if t.modifier}


def write_jsonl(tasks: List[Task], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(t.to_json() + "\n")


def read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

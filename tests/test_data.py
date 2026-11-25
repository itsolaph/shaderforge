import os
import subprocess
import sys

from shaderforge import data as D
from shaderforge import families as F
from shaderforge.reward import MAX_CHARS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COUNTS = {"train": 60, "val": 10, "test_iid": 15, "test_unseen_params": 10, "test_unseen_combo": 10}


def test_determinism():
    a = D.generate_all(COUNTS, seed=3)
    b = D.generate_all(COUNTS, seed=3)
    for s in D.SPLITS:
        assert [t.to_json() for t in a[s]] == [t.to_json() for t in b[s]]
    c = D.generate_all(COUNTS, seed=4)
    assert [t.id for t in a["train"]] != [t.id for t in c["train"]]


def test_determinism_across_processes():
    """No dependence on PYTHONHASHSEED."""
    code = ("from shaderforge import data as D; "
            "t = D.generate_split('train', 20, 0); print(','.join(x.id for x in t))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": str(h), "PYTHONPATH": ROOT}, cwd=ROOT,
                           check=True).stdout
            for h in (1, 2)}
    assert len(outs) == 1


def test_split_disjointness():
    sp = D.generate_all(COUNTS, seed=0)
    ids = {s: {t.id for t in ts} for s, ts in sp.items()}
    assert {s: len(v) for s, v in ids.items()} == COUNTS  # unique within split
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert ids[a].isdisjoint(ids[b]), (a, b)


def test_unseen_combo_never_in_train():
    sp = D.generate_all({**COUNTS, "train": 400, "test_iid": 100}, seed=0)
    seen = D.combos_of(sp["train"]) | D.combos_of(sp["val"]) | D.combos_of(sp["test_iid"])
    held = D.combos_of(sp["test_unseen_combo"])
    assert held and held <= F.HELD_OUT_COMBOS
    assert held.isdisjoint(seen)
    assert seen.isdisjoint(F.HELD_OUT_COMBOS)
    # every held-out task is a combo
    assert all(t.modifier for t in sp["test_unseen_combo"])


def test_unseen_params_outside_train_ranges():
    sp = D.generate_split("test_unseen_params", 80, 0)
    for t in sp:
        for k, v in t.params.items():
            rng = F.FAMILIES[t.family].ranges.get(k)
            if rng:
                lo, hi = rng[0]
                assert not (lo <= v <= hi), (t.family, k, v)


def test_task_content():
    for t in D.generate_split("train", 100, 0):
        assert t.prompt.endswith(t.spec) and "#" in t.spec
        assert "void mainImage" in t.reference and len(t.reference) < MAX_CHARS

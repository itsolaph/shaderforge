import random

import numpy as np
import pytest

from shaderforge import families as F
from shaderforge.reward import is_blank

SIZE = 64


@pytest.mark.parametrize("name", sorted(F.FAMILIES))
@pytest.mark.parametrize("regime", ["train", "unseen"])
def test_family_renders_nonblank(pool, name, regime):
    rng = random.Random(f"{name}-{regime}")
    for _ in range(3):
        s = F.sample_task_params(rng, regime, "single", family=name)
        code = F.build_shader(s["family"], s["params"])
        res = pool.render(code, SIZE)
        assert res.ok, f"{name}: {res.stage} {res.error}"
        assert res.image.shape == (SIZE, SIZE, 3)
        assert not is_blank(res.image), f"{name} rendered blank: {s['params']}"
        assert len(F.build_spec(s["family"], s["params"])) > 40


@pytest.mark.parametrize("mod", sorted(F.MODIFIERS))
def test_modifiers_render(pool, mod):
    rng = random.Random(mod)
    base = rng.choice(sorted(F.FAMILIES))
    p = F.FAMILIES[base].sample(rng)
    mp = F.MODIFIERS[mod].sample(rng)
    res = pool.render(F.build_shader(base, p, mod, mp), SIZE)
    assert res.ok, res.error
    assert not is_blank(res.image)


def test_family_count_and_held_out():
    assert 8 <= len(F.FAMILIES) <= 12
    assert F.HELD_OUT_COMBOS.isdisjoint(F.SEEN_COMBOS)
    assert all(b in F.FAMILIES and m in F.MODIFIERS for b, m in F.HELD_OUT_COMBOS)


def test_sampling_deterministic():
    a = F.sample_task_params(random.Random(5), "train", "mixed")
    b = F.sample_task_params(random.Random(5), "train", "mixed")
    assert a == b


def test_unseen_ranges_disjoint_from_train():
    for fam in F.FAMILIES.values():
        for k, (tr, un) in fam.ranges.items():
            assert tr[1] < un[0] or un[1] < tr[0], f"{fam.name}.{k} overlaps"
    for m in F.MODIFIERS.values():
        for k, (tr, un) in m.ranges.items():
            assert tr[1] < un[0] or un[1] < tr[0], f"{m.name}.{k} overlaps"


def test_rings_geometry(pool):
    """Known semantics: center pixel is c1 (innermost disc), 3 rings of width 1/6."""
    code = F.build_shader("rings", {"n": 3, "c1": "#ff0000", "c2": "#0000ff"})
    img = pool.render(code, 96).image
    assert tuple(img[48, 48]) == (255, 0, 0)
    # ring 1 spans r in [1/6, 2/6] of image height -> 16..32 px from center at 96 px
    assert tuple(img[48, 48 + 24]) == (0, 0, 255)


def test_orientation_origin_bottom_left(pool):
    code = "void mainImage(out vec4 f, in vec2 c) { vec2 uv = c / iResolution.xy; f = vec4(uv.x, uv.y, 0.0, 1.0); }"
    img = pool.render(code, 32).image
    assert img[0, 0, 1] > 200      # top row has high y
    assert img[-1, 0, 1] < 20      # bottom row has low y
    assert img[0, -1, 0] > 200     # right column has high x
    assert img[0, 0, 0] < 20

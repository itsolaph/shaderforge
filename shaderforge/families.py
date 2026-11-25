"""Parametric shader families.

Every family is: a param sampler (with a *train* and an *unseen* numeric regime),
a GLSL body template, and a natural-language spec writer. Modifiers (vignette,
border, scanlines) post-process any base family and are used to build
family combinations for the ``test_unseen_combo`` split.

Conventions shared by every shader (and stated in every prompt):
    uv = fragCoord / iResolution.xy            in [0,1]^2, origin bottom-left
    p  = (fragCoord - 0.5*iResolution.xy) / iResolution.y
                                               centered, units of image height, +y up
Images are square. Everything is static (``ANIMATED`` is a flag for a future v2).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

ANIMATED = False  # v1: static images only, iTime is forbidden by the reward.

TWO_PI = "6.28318530718"
PI = "3.14159265359"


# ----------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------
def fl(x: float) -> str:
    """GLSL float literal (always contains a decimal point)."""
    s = f"{float(x):.4f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    if s in ("-0.0",):
        s = "0.0"
    return s


def gcol(h: str) -> str:
    """GLSL expression for a #rrggbb color."""
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return f"vec3({r}.0, {g}.0, {b}.0) / 255.0"


def _rand_hex(rng: random.Random) -> str:
    return "#%02x%02x%02x" % (rng.randrange(256), rng.randrange(256), rng.randrange(256))


def _l1(a: str, b: str) -> int:
    return sum(abs(int(a[i:i + 2], 16) - int(b[i:i + 2], 16)) for i in (1, 3, 5))


def color_pair(rng: random.Random, min_l1: int = 180) -> Tuple[str, str]:
    while True:
        a, b = _rand_hex(rng), _rand_hex(rng)
        if _l1(a, b) >= min_l1:
            return a, b


# ----------------------------------------------------------------------------
# base families
# ----------------------------------------------------------------------------
# ranges: name -> (train_range, unseen_range); int bounds -> randint, float -> uniform
Range = Tuple[float, float]


@dataclass(frozen=True)
class Family:
    name: str
    ranges: Dict[str, Tuple[Range, Range]]
    body: Callable[[dict], str]
    spec: Callable[[dict], str]
    ncolors: int = 2
    choices: Optional[Dict[str, List[str]]] = None
    finalize: Optional[Callable[[dict, random.Random], None]] = None

    def sample(self, rng: random.Random, regime: str = "train") -> dict:
        idx = 0 if regime == "train" else 1
        params: dict = {}
        for k, pair in self.ranges.items():
            lo, hi = pair[idx]
            if isinstance(lo, int) and isinstance(hi, int):
                params[k] = rng.randint(lo, hi)
            else:
                params[k] = round(rng.uniform(lo, hi), 3)
        for k, opts in (self.choices or {}).items():
            params[k] = rng.choice(opts)
        if self.ncolors == 2:
            params["c1"], params["c2"] = color_pair(rng)
        else:
            params["c1"] = _rand_hex(rng)
        if self.finalize:
            self.finalize(params, rng)
        return params


def _rings_body(p):
    return f"""float w = 0.5 / {fl(p['n'])};
float k = floor(length(p) / w);
col = mod(k, 2.0) < 0.5 ? {gcol(p['c1'])} : {gcol(p['c2'])};"""


def _rings_spec(p):
    return (f"{p['n']} concentric rings centered on the image, alternating {p['c1']} (the innermost disc) "
            f"and {p['c2']}. Every ring is exactly 1/{2 * p['n']} of the image height wide, and the "
            f"pattern keeps repeating into the corners.")


def _checker_body(p):
    return f"""float a = radians({fl(p['angle'])});
mat2 R = mat2(cos(a), -sin(a), sin(a), cos(a));
vec2 q = (R * p + 0.5) * {fl(p['cells'])};
float k = mod(floor(q.x) + floor(q.y), 2.0);
col = k < 0.5 ? {gcol(p['c1'])} : {gcol(p['c2'])};"""


def _checker_spec(p):
    return (f"A checkerboard with {p['cells']} cells per unit square: the grid cell size is 1/{p['cells']} of "
            f"the image height, the grid origin (a cell corner) is at the lower-left corner of the centered "
            f"unit square, and cell (i, j) is {p['c1']} when i + j is even and {p['c2']} otherwise. The whole "
            f"pattern is rotated {p['angle']} degrees counter-clockwise about the image center and extends "
            f"infinitely in all directions.")


def _radial_body(p):
    return f"""float t = clamp(length(p) / {fl(p['radius'])}, 0.0, 1.0);
col = mix({gcol(p['c1'])}, {gcol(p['c2'])}, t);"""


def _radial_spec(p):
    return (f"A radial gradient centered on the image: {p['c1']} at the center blending linearly "
            f"(in RGB) to {p['c2']} at distance {p['radius']} image heights from the center, "
            f"and solid {p['c2']} beyond that.")


def _linear_body(p):
    return f"""float a = radians({fl(p['angle'])});
vec2 d = vec2(cos(a), sin(a));
float t = clamp(0.5 + dot(p, d) / (abs(d.x) + abs(d.y)), 0.0, 1.0);
col = mix({gcol(p['c1'])}, {gcol(p['c2'])}, t);"""


def _linear_spec(p):
    return (f"A linear gradient from {p['c1']} to {p['c2']} blending in RGB along the direction "
            f"{p['angle']} degrees counter-clockwise from +x (0 = left to right, 90 = bottom to top). "
            f"The gradient spans the whole image: t = clamp(0.5 + dot(p, d) / (|d.x| + |d.y|), 0, 1) with "
            f"d = (cos, sin) of that angle, so the two extreme corners are exactly {p['c1']} and {p['c2']}.")


def _dots_body(p):
    return f"""vec2 c = fract(uv * {fl(p['n'])}) - 0.5;
col = length(c) < {fl(p['rf'])} ? {gcol(p['c1'])} : {gcol(p['c2'])};"""


def _dots_spec(p):
    return (f"A polka-dot grid of {p['n']} x {p['n']} cells filling the image. Each cell holds one {p['c1']} "
            f"disc centered in the cell with radius {p['rf']} of the cell width, on a {p['c2']} background.")


def _wobble_body(p):
    return f"""float x = uv.x + {fl(p['amp'])} * sin({TWO_PI} * {fl(p['freq'])} * uv.y);
float k = mod(floor(x * {fl(p['n'])}), 2.0);
col = k < 0.5 ? {gcol(p['c1'])} : {gcol(p['c2'])};"""


def _wobble_spec(p):
    return (f"Wavy vertical stripes: {p['n']} stripes across the image alternating {p['c1']} and {p['c2']}. "
            f"With u, v in [0,1] (v from the bottom), the stripe index is floor({p['n']} * (u + {p['amp']} * "
            f"sin(2*pi*{p['freq']}*v))); even indices are {p['c1']}, odd indices are {p['c2']}.")


def _ngon_body(p):
    if p["kind"] == "polygon":
        return f"""float s = {TWO_PI} / {fl(p['n'])};
float ang = atan(p.y, p.x);
float k = floor(ang / s);
float d = length(p) * cos(ang - (k + 0.5) * s);
col = d < {fl(p['R'])} * cos({PI} / {fl(p['n'])}) ? {gcol(p['c1'])} : {gcol(p['c2'])};"""
    return f"""float s = {TWO_PI} / {fl(p['n'])};
float u = abs(mod(atan(p.y, p.x) + 0.5 * s, s) - 0.5 * s);
vec2 q = length(p) * vec2(cos(u), sin(u));
vec2 O = vec2({fl(p['R'])}, 0.0);
vec2 I = {fl(p['ri'] * p['R'])} * vec2(cos(0.5 * s), sin(0.5 * s));
vec2 e = I - O;
float inside = e.x * (q.y - O.y) - e.y * (q.x - O.x);
col = inside > 0.0 ? {gcol(p['c1'])} : {gcol(p['c2'])};"""


def _ngon_spec(p):
    if p["kind"] == "polygon":
        return (f"A regular {p['n']}-sided polygon centered on the image, filled {p['c1']}, on a {p['c2']} "
                f"background. Circumradius {p['R']} image heights, with one vertex pointing straight right "
                f"(+x). Hard edges.")
    return (f"A {p['n']}-pointed star centered on the image, filled {p['c1']}, on a {p['c2']} background. "
            f"Outer tips at radius {p['R']} image heights (one tip pointing straight right, +x), inner "
            f"vertices at radius {round(p['ri'] * p['R'], 4)} (ratio {p['ri']}) halfway between tips. "
            f"Hard edges.")


def _ngon_final(p, rng):
    if p["kind"] == "polygon":
        p.pop("ri", None)


def _circle_body(p):
    return f"""float m = 1.0 - smoothstep({fl(p['R'] - p['soft'])}, {fl(p['R'] + p['soft'])}, length(p - vec2({fl(p['cx'])}, {fl(p['cy'])})));
col = mix({gcol(p['c2'])}, {gcol(p['c1'])}, m);"""


def _circle_spec(p):
    return (f"A {p['c1']} disc of radius {p['R']} image heights centered at ({p['cx']}, {p['cy']}) "
            f"(p coordinates: image-height units from the image center, +x right, +y up) on a {p['c2']} "
            f"background. The edge is soft: the color blends with smoothstep from radius {round(p['R'] - p['soft'], 4)} "
            f"to {round(p['R'] + p['soft'], 4)}.")


def _vignette_body(p):
    return f"col = {gcol(p['c1'])} * (1.0 - {fl(p['s'])} * smoothstep({fl(p['a'])}, {fl(p['a'] + p['w'])}, length(p)));"


def _vignette_spec(p):
    return (f"A flat {p['c1']} image with a black vignette: the color is multiplied by "
            f"1 - {p['s']} * smoothstep({p['a']}, {round(p['a'] + p['w'], 4)}, r), where r is the distance "
            f"from the image center in image-height units.")


def _vignette_final(p, rng):
    # a near-black base color would make the whole image (almost) constant
    while max(int(p["c1"][i:i + 2], 16) for i in (1, 3, 5)) < 128:
        p["c1"] = _rand_hex(rng)


def _metaball_body(p):
    return f"""vec2 a = p - vec2({fl(p['x1'])}, {fl(p['y1'])});
vec2 b = p - vec2({fl(p['x2'])}, {fl(p['y2'])});
float f = {fl(p['r1'])} * {fl(p['r1'])} / max(dot(a, a), 0.000001) + {fl(p['r2'])} * {fl(p['r2'])} / max(dot(b, b), 0.000001);
col = f >= 1.0 ? {gcol(p['c1'])} : {gcol(p['c2'])};"""


def _metaball_spec(p):
    return (f"Two fused metaballs, {p['c1']} on a {p['c2']} background. Blob A is centered at "
            f"({p['x1']}, {p['y1']}) with radius parameter {p['r1']}; blob B at ({p['x2']}, {p['y2']}) "
            f"with radius parameter {p['r2']} (p coordinates: image-height units from the center, +x right, "
            f"+y up). A pixel is {p['c1']} where r1^2/|p-A|^2 + r2^2/|p-B|^2 >= 1.")


def _metaball_final(p, rng):
    import math
    th = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(0.5, 1.5) * (p["r1"] + p["r2"])
    p["x2"] = round(p["x1"] + dist * math.cos(th), 3)
    p["y2"] = round(p["y1"] + dist * math.sin(th), 3)


def _grid_body(p):
    return f"""vec2 c = fract(uv * {fl(p['n'])});
float m = min(min(c.x, 1.0 - c.x), min(c.y, 1.0 - c.y));
col = m < {fl(p['t'])} * 0.5 ? {gcol(p['c1'])} : {gcol(p['c2'])};"""


def _grid_spec(p):
    return (f"A grid of {p['n']} x {p['n']} cells drawn with {p['c1']} lines on a {p['c2']} background. Lines "
            f"run along every cell border (including the image edges, half of each edge line visible); "
            f"line thickness is {p['t']} of the cell width, centered on the border.")


def _spiral_body(p):
    return f"""float ph = {fl(p['arms'])} * atan(p.y, p.x) / {TWO_PI} + {fl(p['k'])} * length(p);
col = fract(ph) < 0.5 ? {gcol(p['c1'])} : {gcol(p['c2'])};"""


def _spiral_spec(p):
    return (f"A {p['arms']}-armed spiral centered on the image alternating {p['c1']} and {p['c2']}. "
            f"With r the distance from the center in image-height units and theta = atan2(y, x), the phase is "
            f"{p['arms']} * theta / (2*pi) + {p['k']} * r; where fract(phase) < 0.5 the color is {p['c1']}, "
            f"otherwise {p['c2']}.")


FAMILIES: Dict[str, Family] = {f.name: f for f in [
    Family("rings", {"n": ((3, 10), (11, 16))}, _rings_body, _rings_spec),
    Family("checker", {"cells": ((2, 8), (9, 14)), "angle": ((0, 45), (46, 89))}, _checker_body, _checker_spec),
    Family("radial", {"radius": ((0.25, 0.6), (0.62, 0.9))}, _radial_body, _radial_spec),
    Family("linear", {"angle": ((0, 180), (181, 359))}, _linear_body, _linear_spec),
    Family("dots", {"n": ((3, 8), (9, 14)), "rf": ((0.2, 0.42), (0.12, 0.19))}, _dots_body, _dots_spec),
    Family("wobble", {"n": ((4, 12), (13, 20)), "amp": ((0.01, 0.05), (0.06, 0.1)),
                      "freq": ((1, 3), (4, 6))}, _wobble_body, _wobble_spec),
    Family("ngon_star", {"n": ((3, 8), (9, 12)), "R": ((0.2, 0.4), (0.41, 0.48)),
                         "ri": ((0.35, 0.6), (0.2, 0.33))},
           _ngon_body, _ngon_spec, choices={"kind": ["polygon", "star"]}, finalize=_ngon_final),
    Family("circle", {"cx": ((-0.2, 0.2), (0.21, 0.3)), "cy": ((-0.2, 0.2), (0.21, 0.3)),
                      "R": ((0.1, 0.3), (0.31, 0.4)), "soft": ((0.005, 0.04), (0.05, 0.1))},
           _circle_body, _circle_spec),
    Family("vignette", {"s": ((0.3, 0.8), (0.85, 1.0)), "a": ((0.1, 0.3), (0.31, 0.4)),
                        "w": ((0.2, 0.4), (0.1, 0.19))}, _vignette_body, _vignette_spec, ncolors=1,
           finalize=_vignette_final),
    Family("metaball", {"r1": ((0.08, 0.2), (0.21, 0.3)), "r2": ((0.08, 0.2), (0.21, 0.3)),
                        "x1": ((-0.25, 0.25), (0.26, 0.35)), "y1": ((-0.25, 0.25), (0.26, 0.35))},
           _metaball_body, _metaball_spec, finalize=_metaball_final),
    Family("grid", {"n": ((3, 10), (11, 16)), "t": ((0.05, 0.2), (0.02, 0.045))}, _grid_body, _grid_spec),
    Family("spiral", {"arms": ((1, 4), (5, 6)), "k": ((2.0, 6.0), (6.5, 9.0))}, _spiral_body, _spiral_spec),
]}

# ----------------------------------------------------------------------------
# modifiers (applied after a base family)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Modifier:
    name: str
    ranges: Dict[str, Tuple[Range, Range]]
    body: Callable[[dict], str]
    spec: Callable[[dict], str]
    has_color: bool = False

    def sample(self, rng: random.Random, regime: str = "train") -> dict:
        idx = 0 if regime == "train" else 1
        params: dict = {}
        for k, pair in self.ranges.items():
            lo, hi = pair[idx]
            params[k] = rng.randint(lo, hi) if isinstance(lo, int) and isinstance(hi, int) \
                else round(rng.uniform(lo, hi), 3)
        if self.has_color:
            params["bc"] = _rand_hex(rng)
        return params


MODIFIERS: Dict[str, Modifier] = {m.name: m for m in [
    Modifier("vignette", {"s": ((0.3, 0.8), (0.85, 1.0))},
             lambda p: f"col *= 1.0 - {fl(p['s'])} * smoothstep(0.2, 0.7, length(p));",
             lambda p: (f"Finally apply a vignette: multiply the whole color by 1 - {p['s']} * "
                        f"smoothstep(0.2, 0.7, r), where r is the distance from the image center in "
                        f"image-height units.")),
    Modifier("border", {"t": ((0.03, 0.1), (0.11, 0.16))},
             lambda p: (f"float m = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));\n"
                        f"col = m < {fl(p['t'])} ? {gcol(p['bc'])} : col;"),
             lambda p: (f"Finally draw a solid {p['bc']} frame around the image: every pixel closer than "
                        f"{p['t']} (in uv units, i.e. fraction of the image side) to any image edge becomes "
                        f"{p['bc']}."),
             has_color=True),
    Modifier("scanlines", {"n": ((8, 24), (25, 40)), "d": ((0.2, 0.5), (0.55, 0.8))},
             lambda p: f"col *= 1.0 - {fl(p['d'])} * step(0.5, fract(uv.y * {fl(p['n'])}));",
             lambda p: (f"Finally overlay horizontal scanlines: with v in [0,1] from the bottom, multiply the "
                        f"color by 1 - {p['d']} wherever fract(v * {p['n']}) >= 0.5 (so {p['n']} dark bands)."))
]}

# (base, modifier) pairs held out from every training/val/iid split.
HELD_OUT_COMBOS = frozenset({
    ("rings", "vignette"), ("checker", "border"), ("radial", "scanlines"), ("linear", "vignette"),
    ("dots", "border"), ("wobble", "scanlines"), ("ngon_star", "vignette"), ("circle", "scanlines"),
    ("metaball", "border"), ("grid", "vignette"), ("spiral", "border"), ("vignette", "scanlines"),
})
ALL_COMBOS = [(b, m) for b in FAMILIES for m in MODIFIERS]
SEEN_COMBOS = [c for c in ALL_COMBOS if c not in HELD_OUT_COMBOS]
HELD_COMBOS = sorted(HELD_OUT_COMBOS)


# ----------------------------------------------------------------------------
# composition
# ----------------------------------------------------------------------------
def _indent(s: str, n: int = 8) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in s.splitlines())


def build_shader(family: str, params: dict, modifier: Optional[str] = None,
                 mod_params: Optional[dict] = None) -> str:
    fam = FAMILIES[family]
    blocks = [f"    {{\n{_indent(fam.body(params))}\n    }}"]
    if modifier:
        blocks.append(f"    {{\n{_indent(MODIFIERS[modifier].body(mod_params))}\n    }}")
    return (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 uv = fragCoord / iResolution.xy;\n"
        "    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;\n"
        "    vec3 col = vec3(0.0);\n"
        + "\n".join(blocks) + "\n"
        "    fragColor = vec4(col, 1.0);\n"
        "}\n"
    )


def build_spec(family: str, params: dict, modifier: Optional[str] = None,
               mod_params: Optional[dict] = None) -> str:
    s = FAMILIES[family].spec(params)
    if modifier:
        s += " " + MODIFIERS[modifier].spec(mod_params)
    return s


def sample_task_params(rng: random.Random, regime: str = "train", mode: str = "mixed",
                       family: Optional[str] = None) -> dict:
    """Sample (family, params, modifier, mod_params).

    mode: 'mixed'  -> ~60% plain / ~40% seen combo (train/iid/unseen_params)
          'held'   -> always a held-out (base, modifier) combo, train-regime params
          'single' -> plain family
    """
    if mode == "held":
        fam, mod = rng.choice(HELD_COMBOS)
    elif mode == "single" or rng.random() < 0.6:
        fam, mod = family or rng.choice(sorted(FAMILIES)), None
    else:
        fam, mod = rng.choice(SEEN_COMBOS)
    params = FAMILIES[fam].sample(rng, regime)
    mod_params = MODIFIERS[mod].sample(rng, regime) if mod else None
    return {"family": fam, "params": params, "modifier": mod, "mod_params": mod_params}

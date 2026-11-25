"""Reward: compile -> render -> compare with the target image.

    0.0   no code block / forbidden construct / does not compile / render error / timeout
    0.1   compiles but renders a blank / constant image while the target is not
    0.2 + 0.8 * similarity   otherwise, minus a small penalty for > 2500 chars
    similarity = 0.5 * SSIM(luminance) + 0.5 * (1 - min(1, 2 * mean|RGB error|))
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Optional, Union

import numpy as np

from .render import RenderPool, RenderResult, get_pool, load_png

MAX_CHARS = 2500
RENDER_SIZE = 128
SIM_PASS = 0.85

_CODE_RE = re.compile(r"```[ \t]*(?:glsl|GLSL|c|cpp|frag|fragment)?[ \t]*\n(.*?)```", re.S)

# anything that could read external data, break the wrapper, or smuggle time/randomness
_FORBIDDEN = re.compile(
    r"iChannel|iChannelResolution|\btexture\w*\s*\(|\btexelFetch\b|\bsampler\w*|\bimage[1-3]D\b|"
    r"\biTime\b|\biTimeDelta\b|\biFrame\b|\biMouse\b|\biDate\b|\biSampleRate\b|"
    r"#\s*version|#\s*extension|#\s*include|\buniform\b|##|\bdiscard\b|\bgl_FragData\b|\b_sf_out\b|"
    r"\bvoid\s+main\s*\(")


@dataclass
class RewardResult:
    reward: float
    stage: str           # no_code | forbidden | compile | render | timeout | blank | scored
    similarity: float = 0.0
    ssim: float = 0.0
    color: float = 0.0
    n_chars: int = 0
    error: str = ""

    def as_dict(self):
        return asdict(self)


def extract_code(text: str) -> Optional[str]:
    """First fenced code block, or None."""
    m = _CODE_RE.search(text or "")
    return m.group(1).strip("\n") if m else None


def check_forbidden(code: str) -> Optional[str]:
    # strip comments so a mention in a comment is not penalised, but code is checked
    stripped = re.sub(r"//[^\n]*|/\*.*?\*/", " ", code, flags=re.S)
    m = _FORBIDDEN.search(stripped)
    if m:
        return f"forbidden construct: {m.group(0)!r}"
    if not re.search(r"\bvoid\s+mainImage\s*\(", stripped):
        return "missing mainImage"
    return None


# ---------------------------------------------------------------- metrics
def _box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over k x k windows ('valid' region) via integral image."""
    c = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    c[1:, 1:] = a.cumsum(0).cumsum(1)
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def ssim_gray(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    """SSIM of two float images in [0,1], uniform window, mean over valid region."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    n = win * win
    cov_norm = n / (n - 1)
    mu_a, mu_b = _box_mean(a, win), _box_mean(b, win)
    va = cov_norm * (_box_mean(a * a, win) - mu_a ** 2)
    vb = cov_norm * (_box_mean(b * b, win) - mu_b ** 2)
    vab = cov_norm * (_box_mean(a * b, win) - mu_a * mu_b)
    s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def _luma(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float64) / 255.0
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def similarity(img: np.ndarray, target: np.ndarray):
    """-> (similarity, ssim, color_score), each in [0,1]."""
    s = max(0.0, ssim_gray(_luma(img), _luma(target)))
    mae = float(np.abs(img.astype(np.float64) - target.astype(np.float64)).mean() / 255.0)
    col = 1.0 - min(1.0, 2.0 * mae)
    return 0.5 * s + 0.5 * col, s, col


def _spatial_std(img: np.ndarray) -> float:
    """Per-channel spatial std, averaged. A flat colour has 0 even when R, G, B differ."""
    f = img.astype(np.float64)
    if f.ndim == 3:
        return float(f.reshape(-1, f.shape[-1]).std(0).mean())
    return float(f.std())


def is_blank(img: np.ndarray, eps: float = 1.0) -> bool:
    return _spatial_std(img) < eps


@lru_cache(maxsize=4096)
def _load_target(path: str) -> np.ndarray:
    return load_png(path)


def length_penalty(n_chars: int) -> float:
    if n_chars <= MAX_CHARS:
        return 0.0
    return 0.1 * min(1.0, (n_chars - MAX_CHARS) / MAX_CHARS)


# ---------------------------------------------------------------- scoring
def score_render(code: str, res: RenderResult, target: np.ndarray) -> RewardResult:
    n = len(code)
    if not res.ok:
        stage = res.stage if res.stage in ("compile", "render", "timeout") else "render"
        return RewardResult(0.0, stage, n_chars=n, error=res.error[:300])
    img = res.image
    t_std, i_std = _spatial_std(target), _spatial_std(img)
    # "blank" = (near-)constant render against a target that has structure; relative to the target's
    # contrast so a flat mean color plus a faint gradient cannot slip through as non-blank.
    if t_std >= 1.0 and (i_std < 1.0 or i_std < 0.15 * t_std):
        return RewardResult(0.1, "blank", n_chars=n)
    sim, s, col = similarity(img, target)
    r = 0.2 + 0.8 * sim - length_penalty(n)
    return RewardResult(float(min(1.0, max(0.0, r))), "scored", sim, s, col, n)


def _as_target(target: Union[str, np.ndarray]) -> np.ndarray:
    return _load_target(target) if isinstance(target, str) else target


def score_completions(completions, targets, pool: Optional[RenderPool] = None,
                      size: int = RENDER_SIZE):
    """Score text completions against targets (png paths or arrays). Parallel, order preserved."""
    pool = pool or get_pool()
    results: list = [None] * len(completions)
    codes: dict = {}
    for i, text in enumerate(completions):
        code = extract_code(text)
        if code is None:
            results[i] = RewardResult(0.0, "no_code", error="no closed ```glsl block")
            continue
        err = check_forbidden(code)
        if err:
            results[i] = RewardResult(0.0, "forbidden", n_chars=len(code), error=err)
            continue
        codes[i] = code
    uniq = sorted(set(codes.values()))
    rendered = dict(zip(uniq, pool.render_many(uniq, size))) if uniq else {}
    for i, code in codes.items():
        results[i] = score_render(code, rendered[code], _as_target(targets[i]))
    return results


def score_completion(text: str, target, pool: Optional[RenderPool] = None) -> RewardResult:
    return score_completions([text], [target], pool)[0]


def as_block(code: str) -> str:
    return f"```glsl\n{code.strip()}\n```"

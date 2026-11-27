"""Reward: compile -> render -> compare with the target image.

    0.0   no code block / forbidden construct / does not compile / render error / timeout
    0.1   compiles but renders a blank / constant image while the target is not
    0.2 + 0.8 * similarity   otherwise, minus a small penalty for > 2500 chars

Two similarity modes (env ``SF_REWARD``, default ``v2``):
    v1 (runs 1-3): 0.5 * SSIM(luma) + 0.5 * (1 - min(1, 2 * MAE_RGB))
    v2 (run 4+):   0.30 * SSIM(luma) + 0.35 * (1 - min(1, 2 * RMSE_RGB))
                   + 0.35 * multi-scale signed Sobel edge Dice (full + 16x16)
"""
from __future__ import annotations

import os
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
    edge: float = 0.0
    sim_v1: float = 0.0
    n_chars: int = 0
    error: str = ""

    def as_dict(self):
        return asdict(self)


def reward_mode() -> str:
    return os.environ.get("SF_REWARD", "v2").strip().lower()


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


def _sobel_xy(a: np.ndarray):
    gx = (-a[:-2, :-2] + a[:-2, 2:]
          - 2.0 * a[1:-1, :-2] + 2.0 * a[1:-1, 2:]
          - a[2:, :-2] + a[2:, 2:]) / 4.0
    gy = (-a[:-2, :-2] - 2.0 * a[:-2, 1:-1] - a[:-2, 2:]
          + a[2:, :-2] + 2.0 * a[2:, 1:-1] + a[2:, 2:]) / 4.0
    return gx, gy


def _grad_dice(f1: np.ndarray, f2: np.ndarray) -> float:
    """Signed gradient Dice over RGB channels: matches edge location, strength, and polarity."""
    dot = n1 = n2 = 0.0
    for c in range(f1.shape[-1]):
        gx1, gy1 = _sobel_xy(f1[..., c])
        gx2, gy2 = _sobel_xy(f2[..., c])
        dot += float((gx1 * gx2 + gy1 * gy2).sum())
        n1 += float((gx1 * gx1 + gy1 * gy1).sum())
        n2 += float((gx2 * gx2 + gy2 * gy2).sum())
    if n1 + n2 < 1e-8:
        return 1.0
    return float(max(0.0, min(1.0, (2.0 * dot) / (n1 + n2))))


def _downsample_box(f: np.ndarray, target_size: int = 16) -> np.ndarray:
    h, w = f.shape[:2]
    kh, kw = h // target_size, w // target_size
    if kh < 2 or kw < 2:
        return f
    h2, w2 = kh * target_size, kw * target_size
    return f[:h2, :w2].reshape(target_size, kh, target_size, kw, f.shape[-1]).mean(axis=(1, 3))


def edge_similarity(img: np.ndarray, target: np.ndarray) -> float:
    """Multi-scale signed RGB Sobel Dice (full resolution + 16x16 coarse shape)."""
    f1 = img.astype(np.float64) / 255.0
    f2 = target.astype(np.float64) / 255.0
    g_fine = _grad_dice(f1, f2)
    if min(f1.shape[0], f1.shape[1]) >= 32:
        g_coarse = _grad_dice(_downsample_box(f1, 16), _downsample_box(f2, 16))
        return 0.5 * g_fine + 0.5 * g_coarse
    return g_fine


def similarity_details(img: np.ndarray, target: np.ndarray, mode: Optional[str] = None):
    """-> (similarity, ssim, color_score, edge_score, sim_v1), each in [0,1]."""
    m = (mode or reward_mode()).lower()
    s = max(0.0, ssim_gray(_luma(img), _luma(target)))
    f1 = img.astype(np.float64) / 255.0
    f2 = target.astype(np.float64) / 255.0
    mae = float(np.abs(f1 - f2).mean())
    col_mae = 1.0 - min(1.0, 2.0 * mae)
    sim_v1 = 0.5 * s + 0.5 * col_mae
    rmse = float(np.sqrt(np.mean((f1 - f2) ** 2)))
    col_rmse = 1.0 - min(1.0, 2.0 * rmse)
    es = edge_similarity(img, target)
    if m == "v1":
        return sim_v1, s, col_mae, es, sim_v1
    sim_v2 = (30.0 * s + 35.0 * col_rmse + 35.0 * es) / 100.0
    return sim_v2, s, col_rmse, es, sim_v1


def similarity(img: np.ndarray, target: np.ndarray, mode: Optional[str] = None):
    """-> (similarity, ssim, color_score), each in [0,1]."""
    sim, s, col, _, _ = similarity_details(img, target, mode=mode)
    return sim, s, col


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
    sim, s, col, es, sim_v1 = similarity_details(img, target)
    r = 0.2 + 0.8 * sim - length_penalty(n)
    return RewardResult(float(min(1.0, max(0.0, r))), "scored", sim, s, col, es, sim_v1, n)


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

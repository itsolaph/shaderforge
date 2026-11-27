import numpy as np

from shaderforge import families as F
from shaderforge.reward import (as_block, check_forbidden, extract_code, score_completion,
                                score_completions, ssim_gray, similarity, length_penalty)


def _task(n=8, c1="#1e3a8a", c2="#f59e0b"):
    params = {"n": n, "c1": c1, "c2": c2}
    return F.build_shader("rings", params)


def _target(pool, code=None):
    return pool.render(code or _task(), 128).image


def test_ssim_identity_and_difference():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64))
    assert abs(ssim_gray(a, a) - 1.0) < 1e-9
    assert ssim_gray(a, 1 - a) < 0.2


def test_similarity_identity():
    img = (np.random.default_rng(1).random((32, 32, 3)) * 255).astype(np.uint8)
    sim, s, c = similarity(img, img)
    assert sim == 1.0 and s == 1.0 and c == 1.0


def test_extract_code():
    assert extract_code("hi ```glsl\nvoid mainImage(){}\n``` bye") == "void mainImage(){}"
    assert extract_code("no code here") is None
    assert extract_code("```glsl\nunclosed") is None


def test_forbidden_constructs():
    ok = "void mainImage(out vec4 f, in vec2 c) { f = vec4(1.0); }"
    assert check_forbidden(ok) is None
    for bad in ["uniform sampler2D x;", "f = texture(iChannel0, c);", "f = vec4(sin(iTime));",
                "#version 300 es", "void main() {}", "#define A(x) x##y", "f = texelFetch(a,b,0);"]:
        assert check_forbidden(ok + "\n" + bad) is not None, bad
    assert check_forbidden("// iTime is not used\n" + ok) is None
    assert check_forbidden("void foo() {}") == "missing mainImage"


def test_reference_scores_one(pool):
    ref = _task()
    r = score_completion(as_block(ref), _target(pool, ref), pool)
    assert r.stage == "scored" and r.reward > 0.999, r


def test_ordering_reference_perturbed_broken(pool):
    ref, tgt = _task(8), _target(pool, _task(8))
    perturbed = _task(5)
    other_colors = _task(8, "#10b981", "#7c3aed")
    broken = "void mainImage(out vec4 f, in vec2 c) { f = vec4(1.0 }"
    r_ref = score_completion(as_block(ref), tgt, pool).reward
    r_pert = score_completion(as_block(perturbed), tgt, pool).reward
    r_col = score_completion(as_block(other_colors), tgt, pool).reward
    r_bad = score_completion(as_block(broken), tgt, pool).reward
    assert r_ref > r_pert > r_bad
    assert r_ref > r_col > r_bad
    assert r_bad == 0.0


def test_failure_modes(pool):
    tgt = _target(pool)
    blank = "void mainImage(out vec4 f, in vec2 c) { f = vec4(0.3, 0.3, 0.3, 1.0); }"
    tex = "void mainImage(out vec4 f, in vec2 c) { f = texture(iChannel0, c); }"
    res = score_completions([
        "no block at all",
        as_block(blank),
        as_block(tex),
        as_block("void mainImage(out vec4 f, in vec2 c) { f = vec4(undefined_var); }"),
    ], [tgt] * 4, pool)
    assert [r.stage for r in res] == ["no_code", "blank", "forbidden", "compile"]
    assert [r.reward for r in res] == [0.0, 0.1, 0.0, 0.0]


def test_hang_is_killed_and_pool_recovers(pool):
    tgt = _target(pool)
    hang = ("void mainImage(out vec4 f, in vec2 c) { float x = 0.0; "
            "for (int i = 0; i < 2000000000; i++) { x += sin(float(i)); } f = vec4(x); }")
    import time
    t0 = time.monotonic()
    r = pool.render(hang, 128, timeout=2.0)
    # Either the watchdog kills it, or the driver bounds the loop itself
    # (older llvmpipe caps loop iterations at 65535). What matters: no hang.
    assert time.monotonic() - t0 < 10.0
    assert (not r.ok and r.stage in ("timeout", "compile")) or r.ok
    # worker was respawned: normal shader still works
    assert score_completion(as_block(_task()), tgt, pool).reward > 0.99


def test_length_penalty(pool):
    ref = _task()
    padded = ref + "\n// " + "x" * 3000
    tgt = _target(pool, ref)
    assert length_penalty(len(padded)) > 0
    assert score_completion(as_block(padded), tgt, pool).reward < score_completion(as_block(ref), tgt, pool).reward


def test_constant_shader_does_not_beat_reference(pool):
    ref = _task()
    tgt = _target(pool, ref)
    # mean color of the target as a flat color: valid shader, no structure
    m = tgt.reshape(-1, 3).mean(0)
    flat = f"void mainImage(out vec4 f, in vec2 c) {{ f = vec4({m[0]/255:.3f}, {m[1]/255:.3f}, {m[2]/255:.3f}, 1.0); }}"
    r_flat = score_completion(as_block(flat), tgt, pool)
    assert r_flat.reward < 0.7
    # mean color + a faint gradient must not escape the blank check or beat a wrong-but-structured shader
    faint = (f"void mainImage(out vec4 f, in vec2 c) {{ float g = 0.02 * c.y / iResolution.y; "
             f"f = vec4({m[0]/255:.3f} + g, {m[1]/255:.3f} + g, {m[2]/255:.3f} + g, 1.0); }}")
    r_faint = score_completion(as_block(faint), tgt, pool)
    r_pert = score_completion(as_block(_task(5)), tgt, pool)
    assert r_faint.stage == "blank" and r_faint.reward == 0.1
    assert r_faint.reward < r_pert.reward


def test_strict_gate_rejects_vec2_to_float():
    """An old Mesa accepted this; GRPO exploited it in run 2. The glslang gate must refuse it."""
    from shaderforge.render import strict_check
    bad = ("void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
           "    vec2 p = fragCoord / iResolution.xy;\n"
           "    float x = 2.0 * p;\n"
           "    fragColor = vec4(vec3(x), 1.0);\n}\n")
    good = bad.replace("float x = 2.0 * p;", "float x = 2.0 * p.x;")
    assert strict_check(bad) and "glslang" in strict_check(bad)
    assert strict_check(good) is None


def test_shape_reward_rejects_metaball_cross_hack(pool):
    """Run 2 GRPO drew a cross through metaball centres; v1 passed it, v2 must reject it."""
    params = {"x1": 0.126, "y1": -0.239, "r1": 0.189, "x2": -0.138, "y2": -0.453, "r2": 0.101,
              "c1": "#1d7f1c", "c2": "#f701e5"}
    ref = F.build_shader("metaball", params)
    tgt = _target(pool, ref)
    cross = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;\n"
        "    float m = min(min(abs(p.x - 0.126), 1.0 - abs(p.x - (-0.138))),\n"
        "                  min(abs(p.y + 0.239), 1.0 - abs(p.y + 0.453)));\n"
        "    vec3 col = m < 0.189 * 0.189 ? vec3(29.0, 127.0, 28.0) / 255.0 : vec3(247.0, 1.0, 229.0) / 255.0;\n"
        "    fragColor = vec4(col, 1.0);\n}\n"
    )
    r_ref = score_completion(as_block(ref), tgt, pool)
    r_hack = score_completion(as_block(cross), tgt, pool)
    assert r_ref.similarity > 0.999
    assert r_hack.sim_v1 > 0.70
    assert r_hack.similarity < 0.65 and r_hack.edge < 0.30, r_hack


def test_shape_reward_rejects_mirrored_spiral_hack(pool):
    """Run 3 GRPO used abs(atan(p.y, p.x)) mirroring the spiral; v2 must reject it."""
    params = {"arms": 2, "k": 2.985, "c1": "#87791f", "c2": "#1574ce"}
    ref = F.build_shader("spiral", params)
    tgt = _target(pool, ref)
    mirrored = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;\n"
        "    float ph = 2.0 * abs(atan(p.y, p.x)) / 6.28318530718 + 2.985 * length(p);\n"
        "    vec3 col = mod(ph, 1.0) < 0.5 ? vec3(135.0, 121.0, 31.0) / 255.0 : vec3(21.0, 116.0, 206.0) / 255.0;\n"
        "    fragColor = vec4(col, 1.0);\n}\n"
    )
    r_ref = score_completion(as_block(ref), tgt, pool)
    r_hack = score_completion(as_block(mirrored), tgt, pool)
    assert r_ref.similarity > 0.999
    assert r_hack.similarity < 0.80, r_hack

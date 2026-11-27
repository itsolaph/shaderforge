# ShaderForge

Reinforcement learning (GRPO) for a small code model that writes **GLSL fragment shaders from a text description of an image**. The reward is rendered output: compile the shader behind Khronos `glslangValidator`, render it headless, and compare pixels and signed edges with the target. Targets come from a procedurally generated family of reference shaders, so every task has verified ground truth and no scraped data is needed.

**Live WebGL2 contact sheet & playground:** [itsolaph.github.io/shaderforge/](https://itsolaph.github.io/shaderforge/)

> Built for the Vast.ai Fall 2026 Showcase (`#BuiltOnVast`). All four training runs ran on single rented GPUs (RTX 3090 / RTX 4090, **$3.08 total**); data, rendering, reward calibration, eval, and static gallery run on CPU.

## What is in here

| path | what |
|---|---|
| `shaderforge/families.py` | 12 parametric shader families + 3 modifiers (vignette, border, scanlines); each = GLSL template, param sampler (train / unseen ranges), spec writer |
| `shaderforge/render.py` | headless Shadertoy-style renderer (`moderngl`, EGL → Mesa llvmpipe on CPU, GPU driver on GPU boxes); persistent worker subprocess with watchdog respawn; Khronos `glslangValidator` gate (`SF_STRICT=1`); GLSL ES 3.00 so shaders run unchanged in WebGL2 |
| `shaderforge/reward.py` | reward: `0` (no code / forbidden / compile error / timeout) · `0.1` (blank) · `0.2 + 0.8·sim` (`v1`: ½ SSIM + ½ MAE; `v2` shape-aware default: `0.30·SSIM + 0.35·(1−2·RMSE) + 0.35·multi-scale signed RGB Sobel Dice`) |
| `shaderforge/data.py`, `scripts/make_dataset.py` | deterministic splits: `train`, `val`, `test_iid`, `test_unseen_params`, `test_unseen_combo` |
| `scripts/make_family_split.py`, `scripts/probe_warmstart.py` | held-out-family split (`spiral`, `metaball`, `wobble`) and `pass@16` warm-start exploration probe |
| `train/sft.py`, `train/grpo.py`, `train/config.yaml`, `train/full{,2,4}.sh` | SFT warm start, TRL GRPO, and reproducible scripts for Runs 1–4 |
| `eval/evaluate.py` | HF model / OpenAI-compatible endpoint / reference shaders → `eval/results.md` (logs both `v1` and `v2`) |
| `web/` | static live WebGL2 contact sheet (toggle **Run 4** vs **Run 2**) + interactive parameter playground |

### Reward details and anti-hacking
Shaders that mention `iChannel*`, `texture*`, `texelFetch`, samplers, `uniform`, `iTime`/`iMouse`/`iFrame`, `#version`/`#extension`, `##`, `discard` or their own `main()` score `0`. A constant image against a non-constant target scores `0.1`. Every shader must pass Khronos `glslangValidator` (`SF_STRICT=1`). Renders are watchdog-capped against infinite loops, and completions over 2500 characters take a small penalty.

## Four runs, four exploits

Model: `Qwen2.5-Coder-1.5B-Instruct` with LoRA (`r=32`, `alpha=64`). `pass@1` means similarity ≥ 0.85 on a single greedy sample.

### Summary on the 150 held-out-family tasks (`spiral`, `metaball`, `wobble`, strict `glslang` gate)

| run | setup | compile | v1 pixel pass | v2 shape pass | honest pass | what it taught us |
|---|---|---:|---:|---:|---:|---|
| base | Qwen2.5-Coder-1.5B-Instruct | 26.7% | 0.7% | 0.0% | **0.0%** | Rarely emits valid Shadertoy GLSL |
| Run 2 | SFT 2400 + GRPO v1 (loose Mesa) | 69.3% | 48.7% | 36.7% | **36.7%** | Wobble **55/55** honest; spiral Mesa bug (0/46 strict); metaball **cross** hack (18/49 fooled v1) |
| Run 3 | SFT 2400 + GRPO v1 (strict glslang) | 62.7% | 4.7% | 0.7% | **0.7%** | Driver bug & cross dead; overfit SFT (`loss 0.0003`) killed wobble (0/55); 6/7 spirals were `abs(atan)` mirrors |
| **Run 4** | **SFT 480 + GRPO v2 (strict + Sobel)** | **99.3%** | 72.7% | **34.0%** | **32.7%** | Spiral **46/46 (100%, `sim_v2 0.998`)** honest; cross & mirror hacks 0/95; caught **single-blob** metaball shortcut (47/49 rejected by v2) |

### Run 1: original splits are too easy
On `test_iid`, `test_unseen_params`, and `test_unseen_combo`, SFT alone hit 99–100% (`base`: 0–1%). GRPO saw reward `1.0` from step 1 and 75–100% zero-variance groups, so holding out entire unseen shader families became the real benchmark.

### Run 2: held-out families under pixel reward (`v1`, loose Mesa)
Holding out `wobble`, `metaball`, and `spiral` from SFT (`train_sft_a` has only the other 9 families) forced GRPO to learn them from target images alone. Reading all 150 shaders uncovered two exploits beside one real win:
* **Wobble is real (55/55 honest).** SFT never compiled a wobble shader; GRPO discovered `x + a·sin(2π·f·y)` from pixels alone.
* **Metaball is Exploit #1 (the cross hack).** All 49 shaders drew a cross of two bars at the blob centres; 18/49 cleared 0.85 under `v1` pixel similarity.
* **Spiral is Exploit #2 (the Mesa driver hack).** On the rented box, assigning `vec2` to `float` compiled and scored 98%; under Khronos `glslangValidator` and WebGL2, 0/46 compiled. `tests/test_reward.py::test_strict_gate_rejects_vec2_to_float` pins this exact pattern.

### Run 3: strict compiler from the start (`v1`, overfit `sft_2400`)
Rerunning Run 2 with `glslangValidator` active from step 1 killed the driver bug and the metaball cross (0/49), but held-out `pass@1` dropped to **4.7%** (7/150):
* **Wobble collapsed (0/55 compile):** SFT copied `u` and `v` from the prompt without declaring them; every rollout scored 0, leaving zero GRPO gradient.
* **Spiral exposed Exploit #3 (the mirrored `abs(atan)` hack):** 7/46 passed `v1`, but 6 of those 7 used `abs(atan(p.y, p.x))`, mirroring the arms across the x-axis. Only 1/46 was honest (`0.7%` overall).

### Run 4: closing the hacks + fixing the SFT overfitting cliff
Two changes motivated directly by Runs 2–3:
1. **Shape-aware reward (`v2`):** `0.30·SSIM(luma) + 0.35·(1 − min(1, 2·RMSE_RGB)) + 0.35·multi-scale signed RGB Sobel Dice` (128×128 + 16×16). Calibrated offline against Runs 2–3 before renting a GPU: preserves **55/55** genuine Run 2 wobbles (`0.999`) and **1/1** genuine Run 3 spiral (`1.000`), while rejecting **49/49** Run 2 metaball crosses (`max 0.768`) and **6/6** Run 3 mirrored `abs(atan)` spirals (`max 0.812`). Pinned in `tests/test_reward.py`.
2. **SFT warm-start inverted-U ablation:** Run 3 trained SFT for 2400 examples (`loss 0.0003`), crushing entropy before GRPO started. Probing `pass@16` (`T=1.15`, `SF_STRICT=1 SF_REWARD=v2`) on 4 prompts per held-out family revealed a sharp inverted-U:

| SFT samples (steps, `lr=5e-5`) | final loss | wobble compile / std>0 | metaball compile / std>0 | spiral compile / std>0 | active prompts |
|---|---:|---|---|---|---:|
| 0 (`base`) | — | 0.0% · 0/4 | 4.7% · 2/4 | 0.0% · 0/4 | 2 / 12 |
| 80 (10) | 0.246 | 0.0% · 0/4 | 6.2% · 2/4 | 0.0% · 0/4 | 2 / 12 |
| 160 (20) | 0.141 | 4.7% · 2/4 | 7.8% · 3/4 | 1.6% · 1/4 | 6 / 12 |
| 320 (40) | 0.041 | 18.8% · 4/4 (pass 1/4) | 75.0% · 4/4 (pass 1/4) | 1.6% · 1/4 | 9 / 12 |
| **480 (60) · chosen** | **0.0046** | **17.2% · 4/4 (pass 2/4)** | **76.6% · 4/4** | **17.2% · 4/4 (pass 1/4)** | **12 / 12** |
| 640 (80) | 0.019 | 9.4% · 3/4 | 93.8% · 4/4 | 6.2% · 3/4 (pass 1/4) | 10 / 12 |

Starting GRPO from `sft_480` with `G=16`, `T=1.15`, `beta=0.02`, `lr=2.5e-5` (cosine), 80% held-out / 20% seen prompts, 260 steps on one RTX 4090:

| model (`SF_STRICT=1, SF_REWARD=v2`) | compile | mean `sim_v2` | `pass@1 (v2)` | `pass@1 (v1)` | honest |
|---|---:|---:|---:|---:|---:|
| `base` (`Qwen2.5-Coder-1.5B-Instruct`) | 26.7% | 0.005 | 0.0% | 0.7% | 0.0% |
| + `sft_480` (seen families only) | 38.0% | 0.134 | 0.7% | 14.7% | 0.7% |
| **+ GRPO Run 4 (260 steps, images only)** | **99.3%** | **0.756** | **34.0%** | **72.7%** | **32.7%** |

Reading all 150 Run 4 shaders by hand:
* **Spiral is a complete honest win: 46/46 (100.0%, mean `sim_v2 0.998`, min `0.939`).** Every shader compiles in strict `glslang` and WebGL2, uses the true signed `atan(p.y, p.x)` equation (`0/46` `abs(atan)` mirrors), and matches the target pixel-for-pixel.
* **Metaball exposed Exploit #4 (the single-blob shortcut).** With the cross hack dead (`0/49`), GRPO learned to draw Blob A as an exact circle `length(p - c) < r` and omit Blob B. That shortcut fools the old `v1` pixel metric **49/49 (100%)** (`sim_v1 0.858–0.983`), while `v2` catches and rejects **47/49** (median `0.724`; only 2 scanline-heavy frames slip past 0.85).
* **Wobble fixed `u/v` (`54/55` compile, `sin()` in `55/55`),** but converged to a `p.y` phase shift (`2.19 * pi * p.y` instead of `2 * pi * freq * uv.y`), so only `3/55` clear `0.85` under `v2`.
* **Seen-family tradeoff:** `test_iid_a` moved from `89% → 66%` (`v2`) / `91% → 77%` (`v1`) under the aggressive 80/20 held-out RL mix from the lighter `sft_480` warm start.

## Cost receipt

All runs on Vast.ai on-demand single GPUs; every instance destroyed immediately after pulling artefacts.

| run | GPU & host | what | cost |
|---|---|---|---:|
| failed attempts | RTX 3090 24 GB (Ontario, CA, ~$0.22/h) | 0.5B pilots, SFT OOM, stuck host | ≈ $0.44 |
| Run 1 | RTX 3090 24 GB (Ontario, CA, ~$0.22/h) | SFT + GRPO on original splits, full eval | ≈ $0.45 |
| Run 2 | RTX 3090 24 GB (Ontario, CA, ~$0.22/h) | held-out families (`v1`, loose Mesa) | ≈ $0.43 |
| Run 3 | RTX 3090 24 GB (Ontario, CA, ~$0.22/h) | held-out families (`v1`, strict `glslang`) | ≈ $0.42 |
| Run 4 | RTX 4090 24 GB (Utah, US, $0.498/h) | 6-point warm-start probe + 260-step GRPO (`v2` strict + Sobel) | ≈ $1.34 |

**Total: $3.08** of a $10 Vast.ai credit (`$6.92` remaining).

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install moderngl numpy pillow pytest pyyaml
pytest -q                                         # 51 tests (includes driver-hack & reward-hack regressions)
python scripts/make_dataset.py --small            # or omit --small for full 4000/200/300/200/200
python eval/evaluate.py --reference --split test_iid --limit 50
open web/index.html                               # live WebGL2 gallery (Run 4 & Run 2)

bash train/run_on_vast.sh --dry-run               # CPU plumbing check (tiny model, 2 GRPO steps)
bash train/full.sh                                # Run 1 (original splits)
bash train/full2.sh                               # Run 2/3 (set SF_REWARD=v1)
bash train/full4.sh                               # Run 4 (sft_480 + strict glslang + shape reward v2)
```

Dependencies on a fresh Linux GPU box: `apt install glslang-tools libgl1 libgl-dev libegl1 libgl1-mesa-dri libgles2 libgbm1 build-essential`. Set `SF_STRICT=0` to disable the `glslangValidator` gate, or `SF_REWARD=v1` for the legacy pixel metric.

## Prior art & credits

* [`tejadhith/shader`](https://huggingface.co/spaces/tejadhith/shader) – OpenEnv environment (Hugging Face Space), SSIM reward over the shaders21k corpus.
* IR4RL ([arXiv 2609.34587](https://arxiv.org/abs/2609.34587)) – render-progress RL for image-to-SVG/TikZ.
* [`nbardy/shader_benchmark`](https://github.com/nbardy/shader_benchmark) – LLM-judge WGSL benchmark.
* Vipitis' SantaCoder fine-tunes on Shadertoy code ([`Vipitis/santacoder-finetuned-Shadertoys`](https://huggingface.co/Vipitis/santacoder-finetuned-Shadertoys)).

How this differs: text spec → shader (not image-conditioned), a procedurally generated verifiable task family with held-out family, parameter and combination splits, ~$3 single-GPU RL across four controlled runs, an SFT warm-start inverted-U ablation, and a live WebGL2 contact sheet showing the four reward and driver exploits side-by-side with the genuine wins.

## Licenses

Code: MIT (see `LICENSE`). Base model: `Qwen2.5-Coder-1.5B-Instruct` (Apache-2.0). Dependencies keep their own licenses (`moderngl` MIT, `TRL` Apache-2.0, `transformers` Apache-2.0, `peft` Apache-2.0, Mesa MIT, Khronos `glslang` BSD/Apache-2.0). The shader families are original. The project page self-hosts [Archivo](https://github.com/Omnibus-Type/Archivo) by Omnibus-Type (SIL OFL 1.1).

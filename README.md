# ShaderForge

Reinforcement learning (GRPO) for a small code model that writes **GLSL fragment shaders from a text description of an image**. The reward is pixels: compile the shader, render it headless, compare it with the target render. Targets come from a procedurally generated family of reference shaders, so every task has a verified, renderable ground truth and no scraped data is needed.

> Built for the Vast.ai Fall 2026 Showcase (#BuiltOnVast). Training runs on a single rented GPU; everything else (data, rendering, reward, eval, gallery) runs on a CPU-only Linux box.

## What is in here

| path | what |
|---|---|
| `shaderforge/families.py` | 12 parametric shader families + 3 modifiers (vignette, border, scanlines); each = GLSL template, param sampler (train / unseen ranges), spec writer |
| `shaderforge/render.py` | headless Shadertoy-style renderer (moderngl, EGL → Mesa llvmpipe on CPU, GPU driver on GPU boxes); persistent worker subprocess, timeout + respawn; GLSL ES 3.00 so shaders also run in WebGL2 |
| `shaderforge/reward.py` | reward: 0 (no code / forbidden / compile error / timeout) · 0.1 (blank) · 0.2 + 0.8·similarity, similarity = ½ SSIM(luma) + ½ (1 − 2·MAE_RGB) |
| `shaderforge/data.py`, `scripts/make_dataset.py` | deterministic splits: train, val, test_iid, test_unseen_params, test_unseen_combo |
| `train/sft.py`, `train/grpo.py`, `train/config.yaml`, `train/run_on_vast.sh` | SFT warm start, TRL GRPO, one-shot instance script (`--dry-run` for a CPU plumbing check) |
| `eval/evaluate.py` | HF model / OpenAI-compatible endpoint / reference shaders → `eval/results.md` |
| `scripts/make_family_split.py`, `train/full2.sh` | held-out-family experiment (run 2/3) |
| `web/` | static gallery: target vs. SFT vs. GRPO on held-out families, rendered live in WebGL2, plus a parameter playground |

### Reward details and anti-hacking
Shaders that mention `iChannel*`, `texture*`, `texelFetch`, samplers, `uniform`, `iTime`/`iMouse`/`iFrame`, `#version`/`#extension`, `##`, `discard` or their own `main()` score 0. A constant image against a non-constant target scores 0.1. Every shader must pass Khronos `glslangValidator` first. Renders are capped at a few seconds (a hung worker is killed and respawned). Completions over 2500 characters get a small penalty.

### Splits
* `test_iid` – same distribution as train, disjoint task ids.
* `test_unseen_params` – numeric parameters (counts, angles, radii…) drawn from ranges that do not overlap the training ranges.
* `test_unseen_combo` – (base family, modifier) pairs that never appear in train/val/iid (e.g. rings + vignette).

## Results

Two experiments on one rented RTX 3090. Model: Qwen2.5-Coder-1.5B-Instruct with LoRA. pass@1 means similarity ≥ 0.85 on a single greedy sample, and bootstrap 95% CIs are in the eval logs.

### Run 1: the original splits are too easy

| model | test_iid | unseen_params | unseen_combo |
|---|---|---|---|
| base | 1% | 1% | 0% |
| + SFT (4000 examples, 9 min) | 99% | 100% | 100% |
| + SFT + GRPO (200 steps, 84 min) | 99% | 100% | 100% |

SFT solves the benchmark by itself. A spec like "8 concentric rings, #1e3a8a to #f59e0b" maps almost one-to-one onto a template the model has seen 300 times. GRPO had nothing left to learn: the reward was 1.0 from step 1, and 75–100% of groups had zero reward variance, which means zero gradient. Unseen parameter ranges and unseen family/modifier combinations did not make it harder.

### Run 2: families the model never saw code for

`scripts/make_family_split.py` holds out three families (spiral, metaball, wobble). SFT trains only on the other nine. GRPO prompts mix the held-out families with seen ones, and for those it gets only the target image through the reward, never reference code. `test_new_family` has 150 tasks from the held-out families. `test_iid_a` has 100 tasks from the seen families, to check that nothing regresses.

Held-out families, 150 tasks, re-scored on a second machine behind the strict compiler gate (see below):

| model | compile | similarity | pass@1 | wobble | metaball | spiral |
|---|---|---|---|---|---|---|
| base | 26% | 0.017 | 1.3% | 0% | 4% | 0% |
| + SFT (seen families only) | 52% | 0.302 | 0.7% | 0% | 2% | 0% |
| + GRPO, 200 steps | 69% | 0.638 | **48.7%** | **100%** | 37% | 0% |

Seen families (`test_iid_a`): 100% pass@1 after SFT and still 100% after GRPO.

Reading the 150 GRPO shaders one by one changes the picture.

* **Wobble is real.** SFT never wrote a wobble shader that compiled. After GRPO all 55 match, and the code is the actual construction: stripe index computed from `x + a·sin(2π·f·y)`. GRPO discovered it from pixels alone.
* **Metaball is a reward hack.** All 49 GRPO metaball shaders draw a cross of two bars through the blob centres in the right colours. A cross covers roughly the right colour areas, so it scores 0.77–0.89 against two blobs and 37% clear the bar. The model never wrote the field equation. Pixel similarity (½ SSIM + ½ colour MAE) is too forgiving of wrong shapes with the right palette.
* **Spiral is a driver hack.** On the training box the trained model scored 98% on spirals. Its shaders assign a `vec2` to a `float`. The Mesa build on that GPU image accepted it and rendered something spiral-like. Any spec-conforming compiler, including every browser, rejects it. Re-scored, spiral is 0%.

I checked every passing shader by eye. Counting only passes that draw the right shape, the held-out pass rate goes from 0% after SFT (its single pass is a framed vignette that lands on the right colours) to 36.7% after GRPO (all 55 wobble tasks), with no code examples for those families. The base model's two metaball passes do use the field formula, so its honest rate stays at 1.3%.

### The fix: a reference compiler in the reward

`shaderforge/render.py::strict_check` runs every shader through Khronos `glslangValidator` (GLSL ES 3.00) before it is rendered. A shader that the reference front end rejects scores 0 no matter what the local driver does. This closes the spiral hole and makes scores portable between machines. `tests/test_reward.py::test_strict_gate_rejects_vec2_to_float` pins the exact pattern GRPO found. All reference shaders still pass.

### Run 3: same experiment, strict reward

Same split, same 200 steps, glslang gate on during training and eval. Re-scored locally the same way as run 2:

| model | compile | similarity | pass@1 | wobble | metaball | spiral |
|---|---|---|---|---|---|---|
| base | 26% | 0.017 | 1.3% | 0% | 4% | 0% |
| + SFT (seen families only) | 36% | 0.217 | 0% | 0% | 0% | 0% |
| + GRPO, 200 steps | 63% | 0.408 | 4.7% | 0% | 0% | 15% |

Run 2's wobble result did not come back. In run 3 every wobble shader, SFT and GRPO alike, copies `u` and `v` straight from the prompt without declaring them, so all 55 fail to compile and GRPO never gets a non-zero reward to climb from. In run 2 the model happened to declare them early and the rest followed. One run each is not enough to tell how often that happens.

Spiral is now honest but small. 7 of 46 pass. One uses the exact construction from the spec. The other six use `abs(atan(y, x))`, which mirrors the spiral across the x-axis and still lands above 0.85 because the arms nearly line up. Metaball stopped passing (0 of 49), so the cross trick only paid off under the looser compiler.

Seen families held at 99% pass@1 after GRPO. What I take from the three runs: a pixel reward can teach a family from images alone (wobble in run 2, a bit of spiral in run 3), but whether it does depends on the model stumbling into one compiling sample early, and the reward will take a near miss whenever one exists.

The metaball hack is not fixed yet. A shape-aware term in the reward (edge or mask IoU) is the obvious next step.

## Cost receipt

Vast.ai, RTX 3090 24 GB (Ontario, CA), $0.215–0.217/h, on-demand. Every instance was destroyed after its run.

| run | what | cost |
|---|---|---|
| failed attempts | 0.5B pilots, an OOM in SFT, a stuck host | ≈ $0.44 |
| run 1 | SFT + GRPO on original splits, full eval | ≈ $0.45 |
| run 2 | held-out families | ≈ $0.43 |
| run 3 | held-out families, strict reward | ≈ $0.42 |

Total: **$1.74** of a $10 budget.

Practical notes from the runs: Hugging Face downloads from the GPU host crawled at ~0.7 MB/s and stalled, so weights were pulled on a CPU box and copied over SSH. vLLM was not worth installing for a 1.5B model, and plain HF generation in batches of 50 was fine. SFT at batch 8 × 1024 tokens ran out of 24 GB until micro-batching (2) and gradient checkpointing went in.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install moderngl numpy pillow pytest pyyaml
pytest -q
python scripts/make_dataset.py --small            # or without --small for 4000/200/300/200/200
python eval/evaluate.py --reference --split test_iid --limit 50     # ≈ 1.0 upper bound
python scripts/build_gallery.py                   # placeholder gallery -> web/
open web/index.html                               # works from file:// or static hosting

bash train/run_on_vast.sh --dry-run               # CPU plumbing check (tiny model, 2 GRPO steps)
bash train/full.sh                                # run 1 on a GPU box: eval base, SFT, eval, GRPO, eval
bash train/full2.sh                               # run 2/3: held-out-family split, same pipeline
```

The strict reward needs `glslangValidator` (`apt install glslang-tools`). Set `SF_STRICT=0` to turn the gate off.

Evaluating a frontier model: `OPENROUTER_API_KEY=… python eval/evaluate.py --endpoint https://openrouter.ai/api/v1 --api-model <model-id> --label frontier`. The key is read from the environment only.

The renderer needs an EGL implementation (`libEGL`, Mesa llvmpipe is enough) or OSMesa.

## Prior art

* [`tejadhith/shader`](https://huggingface.co/spaces/tejadhith/shader) – OpenEnv environment (Hugging Face Space), SSIM reward over the shaders21k corpus.
* IR4RL ([arXiv 2609.34587](https://arxiv.org/abs/2609.34587)) – render-progress RL for image-to-SVG/TikZ.
* [`nbardy/shader_benchmark`](https://github.com/nbardy/shader_benchmark) – LLM-judge WGSL benchmark.
* Vipitis' SantaCoder fine-tunes on Shadertoy code ([`Vipitis/santacoder-finetuned-Shadertoys`](https://huggingface.co/Vipitis/santacoder-finetuned-Shadertoys)).

How this differs: text spec → shader (not image-conditioned), a procedurally generated verifiable task family with held-out family, parameter and combination splits, a ~$2 single-GPU RL run, and a base vs. SFT vs. GRPO comparison that reports the reward hacks next to the gains. We do not claim to be first.

## Licenses

Code: MIT (see `LICENSE`). Base model: Qwen2.5-Coder-1.5B-Instruct, Apache-2.0. Dependencies keep their own licenses (moderngl MIT, TRL Apache-2.0, Mesa MIT). The shader families are original. The project page self-hosts [Archivo](https://github.com/Omnibus-Type/Archivo) by Omnibus-Type (SIL OFL 1.1).

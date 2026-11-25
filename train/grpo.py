#!/usr/bin/env python
"""GRPO training with the render-and-compare reward (TRL GRPOTrainer).

    python train/grpo.py --config train/config.yaml
    python train/grpo.py --dry-run          # CPU, tiny model, 2 steps: checks the plumbing

NOTE on TRL API drift: GRPOConfig / GRPOTrainer arguments change between TRL releases.
This script therefore builds the config as a dict and keeps only the keys the installed
GRPOConfig accepts (``_accepted``), printing what was dropped. Reward functions receive
``prompts``, ``completions`` and every extra dataset column as keyword arguments
(we pass ``target``); completions are strings for plain prompts and
``[{"role": "assistant", "content": ...}]`` lists for conversational prompts -- both handled.
"""
import argparse
import inspect
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

from shaderforge import data as D  # noqa: E402
from shaderforge.reward import score_completions  # noqa: E402
from shaderforge.render import RenderPool, save_png  # noqa: E402

DRY_MODEL = os.environ.get("SHADERFORGE_DRY_MODEL", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")


def load_config(path, overrides, dry_run):
    cfg = yaml.safe_load(open(path)) if path and os.path.exists(path) else {}
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    if dry_run:
        cfg.update(model=DRY_MODEL, init_from=None, num_generations=2, prompts_per_step=1,
                   max_completion_length=32, max_steps=2, lora=False, bf16=False, use_vllm=False,
                   gradient_checkpointing=False, gallery_every=1, gallery_n=2, save_steps=1000,
                   render_workers=2, output_dir=cfg.get("output_dir", "runs/grpo").rstrip("/") + "_dry")
    return cfg


def _accepted(cls, kwargs):
    ok = set(inspect.signature(cls.__init__).parameters)
    kept = {k: v for k, v in kwargs.items() if k in ok}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        print(f"[grpo] {cls.__name__}: dropped unsupported args {dropped}", file=sys.stderr)
    return kept


def build_rows(data_dir, split, limit, chat):
    rows = []
    for t in D.read_jsonl(os.path.join(data_dir, f"{split}.jsonl"))[:limit]:
        prompt = [{"role": "user", "content": t["prompt"]}] if chat else t["prompt"]
        rows.append({"prompt": prompt, "target": os.path.join(data_dir, t["target"]), "id": t["id"]})
    return rows


def _text(c):
    return c if isinstance(c, str) else c[-1]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "train", "config.yaml"))
    ap.add_argument("--set", action="append", help="override config key=value")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set, a.dry_run)

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    import trl
    from trl import GRPOConfig, GRPOTrainer
    print(f"[grpo] trl {trl.__version__}  torch {torch.__version__}  cuda={torch.cuda.is_available()}")

    has_cuda = torch.cuda.is_available()
    use_vllm = bool(cfg["use_vllm"]) and has_cuda
    if use_vllm:
        try:
            import vllm  # noqa: F401
        except Exception:  # noqa: BLE001
            print("[grpo] vllm not importable -> falling back to HF generate", file=sys.stderr)
            use_vllm = False
    bf16 = bool(cfg["bf16"]) and has_cuda

    out_dir = os.path.join(ROOT, cfg["output_dir"]) if not os.path.isabs(cfg["output_dir"]) else cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    data_dir = cfg["data_dir"] if os.path.isabs(cfg["data_dir"]) else os.path.join(ROOT, cfg["data_dir"])
    model_id = cfg["init_from"] or cfg["model"]

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    chat = bool(tok.chat_template)
    limit = 16 if a.dry_run else None
    train_ds = Dataset.from_list(build_rows(data_dir, cfg["train_split"], limit, chat))
    val_rows = build_rows(data_dir, cfg["val_split"], cfg["gallery_n"], chat)

    dtype = torch.bfloat16 if bf16 else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)

    peft_config = None
    if cfg["lora"]:
        from peft import LoraConfig
        peft_config = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                                 target_modules="all-linear", task_type="CAUSAL_LM")
    lr = cfg["lr_lora"] if cfg["lora"] else cfg["lr_full"]

    ng = int(cfg["num_generations"])
    per_dev = ng  # one prompt's group per micro-batch; accumulate to prompts_per_step prompts
    kw = dict(
        output_dir=out_dir, learning_rate=lr, per_device_train_batch_size=per_dev,
        gradient_accumulation_steps=int(cfg["prompts_per_step"]), num_generations=ng,
        max_completion_length=int(cfg["max_completion_length"]), max_prompt_length=1024,
        beta=float(cfg["beta"]), max_steps=int(cfg["max_steps"]), logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]), bf16=bf16, temperature=float(cfg["temperature"]),
        gradient_checkpointing=bool(cfg["gradient_checkpointing"]), report_to="none", seed=int(cfg["seed"]),
        remove_unused_columns=False, use_vllm=use_vllm, use_cpu=not has_cuda,
        vllm_mode=cfg["vllm_mode"], vllm_gpu_memory_utilization=cfg["vllm_gpu_memory_utilization"],
        log_completions=False)
    args = GRPOConfig(**_accepted(GRPOConfig, kw))

    pool = RenderPool(int(cfg["render_workers"]))
    log_path = os.path.join(out_dir, "train_log.jsonl")
    stats = {"calls": 0}

    def reward_fn(prompts=None, completions=None, target=None, **kwargs):
        res = score_completions([_text(c) for c in completions], list(target), pool)
        stats["calls"] += 1
        n = len(res)
        rec = {"event": "reward", "call": stats["calls"], "t": time.time(), "n": n,
               "compile_rate": sum(r.stage in ("scored", "blank") for r in res) / n,
               "no_code": sum(r.stage == "no_code" for r in res) / n,
               "mean_sim": sum(r.similarity for r in res) / n,
               "mean_reward": sum(r.reward for r in res) / n}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return [r.reward for r in res]

    class Logger(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **k):
            if logs:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"event": "trainer", "step": state.global_step, **{
                        kk: (vv if isinstance(vv, (int, float, str)) else str(vv)) for kk, vv in logs.items()}}) + "\n")

    class Gallery(TrainerCallback):
        """Greedy completions on fixed val prompts every N steps -> json + renders."""

        def on_step_end(self, args_, state, control, model=None, processing_class=None, **k):
            every = int(cfg["gallery_every"])
            if every <= 0 or state.global_step % every != 0:
                return
            try:
                self._snap(state.global_step, model, processing_class or k.get("tokenizer") or tok)
            except Exception as e:  # noqa: BLE001  never kill training for a snapshot
                print(f"[grpo] gallery snapshot failed: {type(e).__name__}: {e}", file=sys.stderr)

        def _snap(self, step, model, tk):
            was_training = model.training
            model.eval()
            prompts = []
            for r in val_rows:
                p = r["prompt"]
                prompts.append(tk.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                               if isinstance(p, list) else p)
            tk.padding_side = "left"
            enc = tk(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=int(cfg["max_completion_length"]), do_sample=False,
                                     pad_token_id=tk.pad_token_id)
            texts = [tk.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen]
            res = score_completions(texts, [r["target"] for r in val_rows], pool)
            gdir = os.path.join(out_dir, "gallery")
            os.makedirs(gdir, exist_ok=True)
            from shaderforge.reward import extract_code
            items = []
            for r, t, rr in zip(val_rows, texts, res):
                code = extract_code(t)
                if code and rr.stage == "scored":
                    img = pool.render(code, 128)
                    if img.ok:
                        save_png(img.image, os.path.join(gdir, f"step{step:05d}_{r['id']}.png"))
                items.append({"id": r["id"], "completion": t, **rr.as_dict()})
            json.dump({"step": step, "items": items}, open(os.path.join(gdir, f"step{step:05d}.json"), "w"), indent=1)
            if was_training:
                model.train()

    trainer = GRPOTrainer(**_accepted(GRPOTrainer, dict(
        model=model, reward_funcs=[reward_fn], args=args, train_dataset=train_ds,
        processing_class=tok, peft_config=peft_config, callbacks=[Logger(), Gallery()])))
    trainer.train()
    final = os.path.join(out_dir, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    if cfg["lora"]:  # also export a merged full model so eval/evaluate.py --model can load it directly
        try:
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(final + "_merged")
            tok.save_pretrained(final + "_merged")
        except Exception as e:  # noqa: BLE001
            print(f"[grpo] LoRA merge failed ({type(e).__name__}: {e}); adapter is in {final}", file=sys.stderr)
    print(f"[grpo] done; saved to {final}; reward calls={stats['calls']}")
    pool.close()


if __name__ == "__main__":
    main()

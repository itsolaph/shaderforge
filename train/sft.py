#!/usr/bin/env python
"""Short SFT warm start on reference shaders (plain PyTorch loop, no TRL dependency).

    python train/sft.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --out runs/sft
    python train/sft.py --dry-run
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shaderforge import data as D  # noqa: E402
from shaderforge.reward import as_block  # noqa: E402

DRY_MODEL = os.environ.get("SHADERFORGE_DRY_MODEL", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")


def encode(tok, prompt, completion, max_len):
    """-> input_ids, labels (prompt tokens masked with -100)."""
    if tok.chat_template:
        ptxt = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
    else:
        ptxt = prompt + "\n"
    p = tok(ptxt, add_special_tokens=False)["input_ids"]
    c = tok(completion + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    ids = (p + c)[:max_len]
    labels = ([-100] * len(p) + c)[:max_len]
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "sft"))
    ap.add_argument("--max-samples", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--micro-batch", type=int, default=2, help="sequences per forward; batch-size is reached by accumulation")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        a.model, a.max_samples, a.batch_size, a.max_len = DRY_MODEL, 8, 2, 512
        a.out = a.out.rstrip("/") + "_dry"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype)
    model.to(dev)
    if a.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, target_modules="all-linear",
                                                 task_type="CAUSAL_LM"))
    if dev == "cuda":
        model.gradient_checkpointing_enable()
        if a.lora:
            model.enable_input_require_grads()
    model.train()

    rows = D.read_jsonl(os.path.join(a.data, f"{a.train_split}.jsonl"))
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.max_samples]
    ex = [encode(tok, r["prompt"], as_block(r["reference"]), a.max_len) for r in rows]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = a.epochs * ((len(ex) + a.batch_size - 1) // a.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.0, 1 - s / max(1, steps)))
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "sft_log.jsonl"), "w")
    step = 0
    for ep in range(a.epochs):
        random.Random(a.seed + ep).shuffle(ex)
        for i in range(0, len(ex), a.batch_size):
            batch = ex[i:i + a.batch_size]
            mb = max(1, min(a.micro_batch, len(batch)))
            tot = sum(sum(1 for t in b[1] if t != -100) for b in batch)
            loss_sum = 0.0
            for j in range(0, len(batch), mb):
                part = batch[j:j + mb]
                L = max(len(b[0]) for b in part)
                ids = torch.tensor([b[0] + [tok.pad_token_id] * (L - len(b[0])) for b in part], device=dev)
                lab = torch.tensor([b[1] + [-100] * (L - len(b[1])) for b in part], device=dev)
                att = torch.tensor([[1] * len(b[0]) + [0] * (L - len(b[0])) for b in part], device=dev)
                n = int((lab[:, 1:] != -100).sum())
                part_loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                (part_loss * n / max(1, tot)).backward()
                loss_sum += float(part_loss) * n
            loss = torch.tensor(loss_sum / max(1, tot))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log.write(json.dumps({"step": step, "loss": float(loss)}) + "\n")
            log.flush()
            if step % 10 == 0 or step == steps:
                print(f"[sft] step {step}/{steps} loss {float(loss):.4f}", flush=True)
    if a.lora:
        model = model.merge_and_unload()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("[sft] saved", a.out)


if __name__ == "__main__":
    main()

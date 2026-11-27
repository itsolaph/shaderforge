#!/usr/bin/env bash
# Run 4: strict glslang gate + shape-aware v2 reward (SSIM + RMSE + multi-scale signed Sobel Dice)
#        + light SFT warm start (preserving exploration entropy) + anti-collapse GRPO.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate

export SF_STRICT=1
export SF_REWARD=v2
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
STEPS="${STEPS:-260}"
SFT_SAMPLES="${SFT_SAMPLES:-480}"
SFT_LR="${SFT_LR:-5e-5}"
GRPO_G="${GRPO_G:-16}"
GRPO_TEMP="${GRPO_TEMP:-1.15}"
GRPO_BETA="${GRPO_BETA:-0.02}"
GRPO_LR="${GRPO_LR:-2.5e-5}"

ev() {
  SF_STRICT=1 SF_REWARD=v2 python eval/evaluate.py --model "$1" --label "$2" --split test_new_family --limit 150 --batch-size 50 \
    --max-new-tokens 512 --workers 16 --no-vllm || return 1
  SF_STRICT=1 SF_REWARD=v2 python eval/evaluate.py --model "$1" --label "$2" --split test_iid_a --limit 100 --batch-size 50 \
    --max-new-tokens 512 --workers 16 --no-vllm
}
stage() { echo; echo "=== $1  $(date -u +%H:%M:%S)"; }

stage split
command -v glslangValidator >/dev/null || { apt-get update -qq >/dev/null; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq glslang-tools >/dev/null; }
glslangValidator --version | head -1 || { echo GLSLANG_MISSING; exit 5; }
python scripts/make_family_split.py --data data --n 150 --workers 16 || { echo SPLIT_FAILED; exit 5; }

python -c '
import json, random
from shaderforge import data as D
HOLDOUT = ("spiral", "metaball", "wobble")
train = D.read_jsonl("data/train.jsonl")
held = [r for r in train if r["family"] in HOLDOUT]
seen = [r for r in train if r["family"] not in HOLDOUT]
rng = random.Random(0)
n_seen = max(1, len(held) // 4)
rl4 = held + rng.sample(seen, min(len(seen), n_seen))
rng.shuffle(rl4)
with open("data/train_rl_v4.jsonl", "w") as f:
    for r in rl4:
        f.write(json.dumps(r, sort_keys=True) + "\n")
print(f"train_rl_v4 {len(rl4)} ({len(held)} held-out + {n_seen} seen)")
'

stage "light SFT (seen families only, samples=${SFT_SAMPLES}, lr=${SFT_LR})"
if [ ! -f runs/sft_d/config.json ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/sft.py --model "$MODEL" --out runs/sft_d --lora \
    --lr "$SFT_LR" --train-split train_sft_a --max-samples "$SFT_SAMPLES" --batch-size 8 --micro-batch 2 --max-len 1024 \
    || { echo SFT_FAILED; exit 6; }
fi

stage "warm-start exploration probe (pass@${GRPO_G})"
python scripts/probe_warmstart.py --model runs/sft_d --label sft_d --k "$GRPO_G" --temp "$GRPO_TEMP" --workers 16

stage "sft_d eval"
[ -f eval/samples_sft_d_test_new_family.jsonl ] || ev runs/sft_d sft_d || { echo EVAL_FAILED; exit 8; }

stage "GRPO Run 4 (strict + shape v2, G=${GRPO_G}, T=${GRPO_TEMP}, beta=${GRPO_BETA}, steps=${STEPS})"
if [ ! -f runs/grpo_d/final_merged/config.json ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/grpo.py --config train/config.yaml \
    --set init_from=runs/sft_d --set model="$MODEL" \
    --set train_split=train_rl_v4 --set max_steps="$STEPS" --set output_dir=runs/grpo_d \
    --set num_generations="$GRPO_G" --set per_device_batch_size="$GRPO_G" --set prompts_per_step=4 \
    --set temperature="$GRPO_TEMP" --set beta="$GRPO_BETA" --set lr_lora="$GRPO_LR" \
    --set lr_scheduler_type=cosine --set warmup_ratio=0.05 \
    --set render_workers=16 --set save_steps=50 --set gallery_every=25 \
    --set max_completion_length=512 --set use_vllm=false \
    || { echo GRPO_FAILED; exit 7; }
fi

stage "grpo_d eval (v2 primary)"
G=runs/grpo_d/final_merged; [ -f $G/config.json ] || G=runs/grpo_d/final
[ -f eval/samples_grpo_d_test_new_family.jsonl ] || ev "$G" grpo_d || { echo EVAL_FAILED; exit 8; }

stage "base_d eval (v2 apples-to-apples)"
[ -f eval/samples_base_d_test_new_family.jsonl ] || ev "$MODEL" base_d || { echo EVAL_FAILED; exit 8; }

stage pack
tar czf runs/full4.tgz eval/results.md eval/results.json eval/samples_*_d_* runs/sft_d/sft_log.jsonl \
  runs/grpo_d/train_log.jsonl runs/grpo_d/gallery data/test_new_family.jsonl 2>/dev/null
ls -la runs/full4.tgz
echo FULL4_DONE

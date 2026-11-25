#!/usr/bin/env bash
# Run 2: held-out shader families. SFT never sees spiral/metaball/wobble code; GRPO gets their target
# images only. Question: does render-reward RL teach families that supervised data never covered?
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
STEPS="${STEPS:-200}"
ev() { python eval/evaluate.py --model "$1" --label "$2" --split test_new_family --limit 150 --batch-size 50 \
         --max-new-tokens 512 --workers 16 --no-vllm || return 1
       python eval/evaluate.py --model "$1" --label "$2" --split test_iid_a --limit 100 --batch-size 50 \
         --max-new-tokens 512 --workers 16 --no-vllm; }
stage() { echo; echo "=== $1  $(date -u +%H:%M:%S)"; }

stage split
command -v glslangValidator >/dev/null || { apt-get update -qq >/dev/null; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq glslang-tools >/dev/null; }
glslangValidator --version | head -1 || { echo GLSLANG_MISSING; exit 5; }
python scripts/make_family_split.py --data data --n 150 --workers 16 || { echo SPLIT_FAILED; exit 5; }

stage "base eval"
[ -f eval/samples_base_c_test_new_family.jsonl ] || ev "$MODEL" base_c || { echo EVAL_FAILED; exit 8; }

stage "SFT (seen families only)"
if [ ! -f runs/sft/config.json ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/sft.py --model "$MODEL" --out runs/sft --lora \
    --lr 1e-4 --train-split train_sft_a --max-samples 2400 --batch-size 8 --micro-batch 2 --max-len 1024 \
    || { echo SFT_FAILED; exit 6; }
fi

stage "sft eval"
[ -f eval/samples_sft_c_test_new_family.jsonl ] || ev runs/sft sft_c || { echo EVAL_FAILED; exit 8; }

stage "GRPO (held-out families: images only)"
if [ ! -f runs/grpo/final_merged/config.json ]; then
  python train/grpo.py --config train/config.yaml --set init_from=runs/sft --set model="$MODEL" \
    --set train_split=train_rl --set max_steps="$STEPS" --set output_dir=runs/grpo --set render_workers=16 \
    --set save_steps=50 --set gallery_every=25 --set max_completion_length=512 --set use_vllm=false \
    || { echo GRPO_FAILED; exit 7; }
fi

stage "grpo eval"
G=runs/grpo/final_merged; [ -f $G/config.json ] || G=runs/grpo/final
[ -f eval/samples_grpo_c_test_new_family.jsonl ] || ev "$G" grpo_c || { echo EVAL_FAILED; exit 8; }

stage pack
tar czf runs/full.tgz eval/results.md eval/results.json eval/samples_*_c_* runs/sft/sft_log.jsonl \
  runs/grpo/train_log.jsonl runs/grpo/gallery data/test_new_family.jsonl 2>/dev/null
ls -la runs/full.tgz
echo FULL_DONE

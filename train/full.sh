#!/usr/bin/env bash
# Full run: base eval -> SFT warm start -> eval -> GRPO -> eval. Each stage is skipped if its output exists.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
STEPS="${STEPS:-200}"
EV="--split all --limit 100 --batch-size 50 --max-new-tokens 512 --workers 16 --no-vllm"
stage() { echo; echo "=== $1  $(date -u +%H:%M:%S)"; }

stage deps
python - <<'PY'
import torch, transformers, trl
try:
    import vllm; v = vllm.__version__
except Exception as e:
    v = f"unavailable ({type(e).__name__}: {e})"
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| transformers", transformers.__version__,
      "| trl", trl.__version__, "| vllm", v)
PY
python -m pytest -q tests > pytest.log 2>&1 || { tail -20 pytest.log; echo PYTEST_FAILED; exit 4; }
tail -1 pytest.log

stage "base eval"
[ -f eval/samples_base15_test_iid.jsonl ] || python eval/evaluate.py --model "$MODEL" --label base15 $EV

stage SFT
if [ ! -f runs/sft/config.json ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/sft.py --model "$MODEL" --out runs/sft --lora --lr 1e-4 --max-samples 2400 --batch-size 8 --micro-batch 2 --max-len 1024 || { echo SFT_FAILED; exit 6; }
fi

stage "sft eval"
[ -f eval/samples_sft15_test_iid.jsonl ] || python eval/evaluate.py --model runs/sft --label sft15 $EV

stage GRPO
if [ ! -f runs/grpo/final_merged/config.json ]; then
  python train/grpo.py --config train/config.yaml --set init_from=runs/sft --set model="$MODEL" \
    --set max_steps="$STEPS" --set output_dir=runs/grpo --set render_workers=16 --set save_steps=50 \
    --set gallery_every=25 --set max_completion_length=512 --set use_vllm=false \
  || { echo "GRPO with vLLM failed -> retrying with HF generate"; rm -rf runs/grpo;
       python train/grpo.py --config train/config.yaml --set init_from=runs/sft --set model="$MODEL" \
         --set max_steps="$STEPS" --set output_dir=runs/grpo --set render_workers=16 --set save_steps=50 \
         --set gallery_every=25 --set max_completion_length=512 --set use_vllm=false || { echo GRPO_FAILED; exit 7; }; }
fi

stage "grpo eval"
G=runs/grpo/final_merged; [ -f $G/config.json ] || G=runs/grpo/final
[ -f eval/samples_grpo15_test_iid.jsonl ] || python eval/evaluate.py --model "$G" --label grpo15 $EV || { echo EVAL_FAILED; exit 8; }

stage pack
tar czf runs/full.tgz eval/results.md eval/results.json eval/samples_*15_* runs/sft/sft_log.jsonl \
  runs/grpo/train_log.jsonl runs/grpo/gallery 2>/dev/null
ls -la runs/full.tgz
echo FULL_DONE

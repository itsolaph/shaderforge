#!/usr/bin/env bash
# Short GPU pilot: base eval + a few GRPO steps, to measure step time and check the reward moves.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"
STEPS="${STEPS:-20}"
export HF_HUB_ENABLE_HF_TRANSFER=0
echo "== env"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$(id -u)" = 0 ] && ! ldconfig -p | grep -q libEGL_mesa; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 libegl-mesa0 libgles2 libgl1-mesa-dri glslang-tools >/dev/null
fi
python3 -m venv --system-site-packages .venv && . .venv/bin/activate
pip install -q moderngl numpy pillow pytest pyyaml "transformers>=5.20" "trl>=1.18" peft accelerate datasets 2>&1 | tail -2
python - <<'PY'
import moderngl
c = moderngl.create_standalone_context(backend="egl")
print("GL:", c.info["GL_RENDERER"])
PY
python -m pytest -q tests > pytest.log 2>&1 || { tail -20 pytest.log; echo PYTEST_FAILED; exit 4; }; tail -1 pytest.log
python scripts/make_dataset.py --out data || { echo DATASET_FAILED; exit 5; }
t0=$(date +%s)
python eval/evaluate.py --model "$MODEL" --label base --split test_iid --limit 60
echo "base_eval_s=$(( $(date +%s) - t0 ))"
t0=$(date +%s)
python train/grpo.py --config train/config.yaml --set model="$MODEL" --set max_steps="$STEPS" \
  --set output_dir=runs/pilot --set use_vllm=false --set save_steps=1000 --set gallery_every=10
echo "grpo_s=$(( $(date +%s) - t0 )) steps=$STEPS"
python eval/evaluate.py --model runs/pilot/final --label pilot --split test_iid --limit 60 || true
tar czf runs/pilot.tgz eval/results.md eval/results.json eval/samples_* runs/pilot/train_log.jsonl runs/pilot/gallery 2>/dev/null
echo PILOT_DONE

#!/usr/bin/env bash
# Set up a fresh Vast.ai PyTorch instance and run SFT -> GRPO -> eval.
#
#   bash train/run_on_vast.sh              # full run (GPU)
#   bash train/run_on_vast.sh --dry-run    # CPU plumbing check: tiny model, 2 GRPO steps
#
# Optional env: MODEL (default Qwen/Qwen2.5-Coder-1.5B-Instruct), SFT_SAMPLES, GRPO_STEPS,
#               HF_REPO + HF_TOKEN (push artefacts to the Hugging Face Hub; never hardcode tokens).
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
SFT_SAMPLES="${SFT_SAMPLES:-1500}"
GRPO_STEPS="${GRPO_STEPS:-300}"

echo "== [1/6] environment"
# Software GL fallback for headless boxes (root only; harmless if already installed).
if [ "$(id -u)" = "0" ] && command -v apt-get >/dev/null 2>&1 && ! ldconfig -p | grep -q libEGL.so.1; then
  apt-get update -qq && apt-get install -y -qq libegl1 libgl1-mesa-dri libgles2 libgbm1 >/dev/null
fi
if [ ! -d .venv ]; then
  python3 -m venv --system-site-packages .venv   # reuse the image's preinstalled torch
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if ! python -c "import torch" 2>/dev/null; then
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -q -r train/requirements.txt pytest
if [ "$DRY" = "0" ] && python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  pip install -q vllm || echo "vllm install failed; GRPO will use HF generate"
fi

echo "== [2/6] renderer self-test"
python -m pytest -q tests/test_families.py -k "rings_geometry or orientation"

echo "== [3/6] dataset"
if [ "$DRY" = "1" ]; then
  python scripts/make_dataset.py --small --out data
elif [ ! -f data/train.jsonl ]; then
  python scripts/make_dataset.py --out data
fi

if [ "$DRY" = "1" ]; then
  echo "== [4/6] SFT (dry)"
  python train/sft.py --dry-run
  echo "== [5/6] GRPO (dry: 2 steps, tiny model)"
  python train/grpo.py --dry-run
  echo "== [6/6] eval (reference sanity)"
  python eval/evaluate.py --reference --split test_iid --limit 5 --out-dir runs/dry_eval
  echo "DRY RUN OK"
  exit 0
fi

echo "== [4/6] base model eval + SFT warm start"
python eval/evaluate.py --model "$MODEL" --label base
python train/sft.py --model "$MODEL" --out runs/sft --max-samples "$SFT_SAMPLES"
python eval/evaluate.py --model runs/sft --label sft

echo "== [5/6] GRPO"
python train/grpo.py --config train/config.yaml \
  --set init_from=runs/sft --set max_steps="$GRPO_STEPS" --set output_dir=runs/grpo
FINAL=runs/grpo/final
[ -d runs/grpo/final_merged ] && FINAL=runs/grpo/final_merged
python eval/evaluate.py --model "$FINAL" --label grpo

echo "== [6/6] artefacts"
python scripts/build_gallery.py --base-samples eval/samples_base_test_iid.jsonl \
  --trained-samples eval/samples_grpo_test_iid.jsonl --out web || echo "gallery build failed"
tar czf runs/artifacts.tgz eval/results.md eval/results.json runs/grpo/train_log.jsonl runs/grpo/gallery web 2>/dev/null || true
if [ -n "${HF_REPO:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
  pip install -q huggingface_hub
  python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["HF_REPO"]
api.create_repo(repo, exist_ok=True)
api.upload_folder(folder_path="runs/grpo", repo_id=repo, path_in_repo="grpo")
api.upload_file(path_or_fileobj="runs/artifacts.tgz", path_in_repo="artifacts.tgz", repo_id=repo)
print("pushed to", repo)
PY
else
  echo "HF_REPO/HF_TOKEN not set: artefacts left in runs/ (scp runs/artifacts.tgz out before destroying the instance)"
fi
echo "DONE"

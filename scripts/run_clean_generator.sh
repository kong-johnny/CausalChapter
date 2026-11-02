#!/usr/bin/env bash
# Run in the intended environment (on AIS, source the PPU SDK and venv first).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CC_CLEAN_GENERATOR_ROOT:?Set a new task-owned experiment directory}"
: "${CC_LCDS_CHECKPOINT:?Set the frozen released LCDS checkpoint directory}"
: "${CC_FULL_VIS_CACHE:?Set the complete utterance feature cache}"
: "${CC_BASE_MODEL:?Set a complete fresh local Qwen base, not an adapter}"
: "${CC_LCDS_DEV_BOUNDARIES:?Set the frozen predicted dev boundary JSONL}"
: "${CC_LCDS_DEV_TIMES:?Set the matching real-time dev chapter JSONL}"
if [[ "$CC_CLEAN_GENERATOR_ROOT" != /* || "$CC_CLEAN_GENERATOR_ROOT" == / ]]; then
  echo "Require an absolute, task-specific output path" >&2
  exit 2
fi
if [[ -e "$CC_CLEAN_GENERATOR_ROOT/train_predictions" || -e "$CC_CLEAN_GENERATOR_ROOT/train/training_config.json" ]]; then
  echo "Experiment already started; inspect its process and logs, do not duplicate" >&2
  exit 2
fi
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 CC_SEMANTIC_BACKEND=hashing
python -u -m segment.infer_lcds --checkpoint "$CC_LCDS_CHECKPOINT" \
  --text "$ROOT/data/avlecture/train.jsonl" --visual-cache "$CC_FULL_VIS_CACHE" \
  --output "$CC_CLEAN_GENERATOR_ROOT/train_predictions" --device cuda
args=(--data-dir "$ROOT/data/avlecture" --base-model "$CC_BASE_MODEL"
      --visual-cache "$CC_FULL_VIS_CACHE"
      --train-boundaries "$CC_CLEAN_GENERATOR_ROOT/train_predictions/boundaries.jsonl"
      --train-times "$CC_CLEAN_GENERATOR_ROOT/train_predictions/chapter_times.jsonl"
      --dev-boundaries "$CC_LCDS_DEV_BOUNDARIES" --dev-times "$CC_LCDS_DEV_TIMES"
      --fusion text_only --epochs 3 --lr 2e-5 --rank 32 --grad-accum 4
      --max-length 4096 --input-tokens 8192 --beams 3 --topk 3 --seed 42
      --caption-matching lexical --device cuda)
python -u -m generate.train_clean "${args[@]}" --smoke-steps 1 \
  --output "$CC_CLEAN_GENERATOR_ROOT/smoke"
python -u -m generate.train_clean "${args[@]}" --output "$CC_CLEAN_GENERATOR_ROOT/train"
echo "CLEAN_GENERATOR_PIPELINE_COMPLETE; no inherited adapter, test or oracle generation"

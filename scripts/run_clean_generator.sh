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
if [[ -e "$CC_CLEAN_GENERATOR_ROOT/train/training_config.json" ]]; then
  echo "Experiment already started; inspect its process and logs, do not duplicate" >&2
  exit 2
fi
export USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 CC_SEMANTIC_BACKEND=hashing
if [[ "${CC_REUSE_TRAIN_PREDICTIONS:-0}" == 1 ]]; then
  # Explicit phase recovery, never optimizer/checkpoint resume. The trainer
  # audits split IDs, lengths, complete real timelines and input hashes.
  test -s "$CC_CLEAN_GENERATOR_ROOT/train_predictions/boundaries.jsonl"
  test -s "$CC_CLEAN_GENERATOR_ROOT/train_predictions/chapter_times.jsonl"
  echo "REUSING_PREPARED_TRAIN_PREDICTIONS; fresh generator initialization"
else
  if [[ -e "$CC_CLEAN_GENERATOR_ROOT/train_predictions" ]]; then
    echo "Prepared inputs already exist; inspect prior process before explicit recovery" >&2
    exit 2
  fi
  python -u -m segment.infer_lcds --checkpoint "$CC_LCDS_CHECKPOINT" \
    --text "$ROOT/data/avlecture/train.jsonl" --visual-cache "$CC_FULL_VIS_CACHE" \
    --output "$CC_CLEAN_GENERATOR_ROOT/train_predictions" --device cuda
fi
args=(--data-dir "$ROOT/data/avlecture" --base-model "$CC_BASE_MODEL"
      --visual-cache "$CC_FULL_VIS_CACHE"
      --train-boundaries "$CC_CLEAN_GENERATOR_ROOT/train_predictions/boundaries.jsonl"
      --train-times "$CC_CLEAN_GENERATOR_ROOT/train_predictions/chapter_times.jsonl"
      --dev-boundaries "$CC_LCDS_DEV_BOUNDARIES" --dev-times "$CC_LCDS_DEV_TIMES"
      --fusion text_only --epochs 3 --lr 2e-5 --rank 32 --grad-accum 4
      --max-length 4096 --input-tokens 8192 --beams 3 --topk 3 --seed 42
      --caption-matching lexical --device cuda)
python -u -m generate.train_clean "${args[@]}" --smoke-steps 1 \
  --output "${CC_SMOKE_OUTPUT:-$CC_CLEAN_GENERATOR_ROOT/smoke}"
python -u -m generate.train_clean "${args[@]}" --output "$CC_CLEAN_GENERATOR_ROOT/train"
echo "CLEAN_GENERATOR_PIPELINE_COMPLETE; no inherited adapter, test or oracle generation"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export USE_TF=0
export USE_FLAX=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

BASE_MODEL="${CC_BASE_MODEL:-$ROOT/models/Qwen2.5-3B-Instruct}"
CHECKPOINT="${CC_SEG_CHECKPOINT:-$ROOT/checkpoints/segment_qwen2p5_3b_endpos_z035}"
TEXT="${CC_TEST_TEXT:-$ROOT/data/avlecture/test.jsonl}"
GROUND_TRUTH="${CC_TEST_GT:-$ROOT/data/avlecture/test_subheading_abstract.jsonl}"
OUTPUT_DIR="${CC_SEG_OUTPUT_DIR:-$ROOT/outputs/segment_reproduce}"

python3 -m segment.predict \
  --base-model "$BASE_MODEL" \
  --checkpoint "$CHECKPOINT" \
  --text "$TEXT" \
  --ground-truth "$GROUND_TRUTH" \
  --output "$OUTPUT_DIR/test_boundaries.jsonl" \
  --metrics-output "$OUTPUT_DIR/metrics.json" \
  --threshold 0.35 \
  --min-gap 6

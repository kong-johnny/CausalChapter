#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export USE_TF=0 USE_FLAX=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python3 scripts/verify_artifacts.py
python3 scripts/prepare_eval_features.py \
  --archive "$ROOT/data/avlecture/lcds_visual_features.tar.gz" --destination "$ROOT/data/lcds"
VISUAL_CACHE="$ROOT/data/lcds/vis2d_cache/avlecture"
SEGMENT_OUTPUT="${CC_SEG_OUTPUT_DIR:-$ROOT/outputs/lcds_inference}"
python3 -m segment.infer_lcds \
  --checkpoint "$ROOT/checkpoints/lcds_longformer_reconstruction" \
  --text "$ROOT/data/avlecture/test.jsonl" --visual-cache "$VISUAL_CACHE" \
  --output "$SEGMENT_OUTPUT"
export CC_TEST_PRED_BOUNDARY="$SEGMENT_OUTPUT/boundaries.jsonl"
export CC_CHAPTER_TIMES="$SEGMENT_OUTPUT/chapter_times.jsonl"
export CC_VIS_CACHE_DIR="$VISUAL_CACHE"
export CC_EVAL_PROTOCOL="strict" CC_EVAL_SCOPE="pred"
export CC_EVAL_SAVE_DIR="${CC_EVAL_SAVE_DIR:-$ROOT/outputs/lcds_generate_reproduced}"
bash scripts/reproduce_best.sh

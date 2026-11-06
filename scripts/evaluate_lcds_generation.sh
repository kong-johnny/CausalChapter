#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CC_FULL_VIS_CACHE:?Set CC_FULL_VIS_CACHE to the complete utterance visual cache}"
LCDS_OUTPUT="${CC_LCDS_OUTPUT:-$ROOT/outputs/lcds_reconstruction}"
export CC_TEST_PRED_BOUNDARY="$LCDS_OUTPUT/test_boundaries.jsonl"
export CC_CHAPTER_TIMES="$LCDS_OUTPUT/test_chapter_times.jsonl"
export CC_VIS_CACHE_DIR="$CC_FULL_VIS_CACHE"
export CC_EVAL_PROTOCOL="strict"
export CC_EVAL_SCOPE="both"
export CC_EVAL_SAVE_DIR="${CC_EVAL_SAVE_DIR:-$ROOT/outputs/lcds_generate_strict}"
bash "$ROOT/scripts/reproduce_legacy_best.sh"

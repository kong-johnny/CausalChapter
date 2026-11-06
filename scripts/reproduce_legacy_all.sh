#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash scripts/verify_artifacts.sh --profile all
python3 scripts/prepare_eval_features.py
bash scripts/reproduce_segmentation.sh

export CC_TEST_PRED_BOUNDARY="${CC_TEST_PRED_BOUNDARY:-$ROOT/outputs/segment_reproduce/test_boundaries.jsonl}"
bash scripts/reproduce_legacy_best.sh

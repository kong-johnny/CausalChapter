#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export USE_TF=0
export USE_FLAX=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

python3 -m segment.train_lcds train \
  --base-model "${CC_LONGFORMER_MODEL:-allenai/longformer-base-4096}" \
  --data-dir "${CC_DATA_DIR:-$ROOT/data/avlecture}" \
  --visual-cache "${CC_TRAIN_VIS_CACHE:-$ROOT/data/avlecture/vis2d_cache/avlecture}" \
  --output "${CC_LCDS_OUTPUT:-$ROOT/outputs/lcds_reconstruction}" \
  --epochs "${CC_LCDS_EPOCHS:-10}" \
  --lr 5e-5 --grad-accum 4 --window-size 5 --seed 42

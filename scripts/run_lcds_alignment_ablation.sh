#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CC_LCDS_INIT:?Set CC_LCDS_INIT to the frozen first-run best.pt}"
: "${CC_FULL_VIS_CACHE:?Set CC_FULL_VIS_CACHE to the complete visual cache}"
: "${CC_LONGFORMER_MODEL:?Set CC_LONGFORMER_MODEL to the pinned base initialization}"
: "${CC_ABLATION_OUTPUT:?Set a new CC_ABLATION_OUTPUT directory}"
export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

for variant in global local2; do
  output="$CC_ABLATION_OUTPUT/$variant"
  if [[ -e "$output/run_config.json" ]]; then
    echo "Refusing to overwrite prior experiment: $output" >&2
    exit 2
  fi
  extra=()
  if [[ "$variant" == local2 ]]; then extra+=(--visual-radius 2); fi
  python -u -m segment.train_lcds train \
    --base-model "$CC_LONGFORMER_MODEL" --data-dir "$ROOT/data/avlecture" \
    --visual-cache "$CC_FULL_VIS_CACHE" --output "$output" \
    --warm-start "$CC_LCDS_INIT" --epochs 3 --lr 1e-5 --grad-accum 4 \
    --window-size 5 --seed 42 --selection-anchor middle "${extra[@]}"
  python -u -m segment.train_lcds evaluate --split dev \
    --data-dir "$ROOT/data/avlecture" --visual-cache "$CC_FULL_VIS_CACHE" --output "$output"
  python scripts/tune_lcds_dev.py --text "$ROOT/data/avlecture/dev.jsonl" \
    --scores "$output/dev_scores.json" --output "$output/dev_decoding_tuning.json"
done
echo "ALIGNMENT_ABLATION_COMPLETE; only train/dev were loaded; no test selection"

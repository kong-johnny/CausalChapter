#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export CC_BASE_MODEL="${CC_BASE_MODEL:-$ROOT/models/Qwen2.5-3B-Instruct}"
export CC_CHECKPOINT_DIR="${CC_CHECKPOINT_DIR:-$ROOT/checkpoints/qwen2p5_3b_crossattn_clip_best}"
export CC_TEST_PRED_BOUNDARY="${CC_TEST_PRED_BOUNDARY:-$ROOT/boundaries/boundaries_12_thr0.35_gap6.jsonl}"
export CC_EVAL_SAVE_DIR="${CC_EVAL_SAVE_DIR:-$ROOT/outputs/reproduce_best}"
export CC_VIS_CACHE_DIR="${CC_VIS_CACHE_DIR:-$ROOT/data/avlecture/vis2d_cache/avlecture}"

export CC_FUSION_MODE="cross_attn"
export CC_NUM_QUERY_TOKENS="8"
export CC_VIS_DIM="512"
export CC_CAUSAL_MODE="light"
export CC_CAUSAL_TOPK="3"
export CC_RETRIEVAL_NEAR_SIZE="3"
export CC_RETRIEVAL_SEMANTIC_SIZE="10"
export CC_RETRIEVAL_INTERVENTION_TOKENS="16"
export CC_RETRIEVAL_VARIANT_MICROBATCH="2"
export CC_USE_FEWSHOT="0"
export CC_EVAL_SCOPE="pred"
export CC_GEN_NUM_BEAMS="3"
export CC_GEN_REPETITION_PENALTY="1.005"
export CC_GEN_NO_REPEAT_NGRAM_SIZE="3"

mkdir -p "$CC_EVAL_SAVE_DIR"
python3 -m generate.eval_env

echo
echo "Saved to: $CC_EVAL_SAVE_DIR"
cat "$CC_EVAL_SAVE_DIR/eval_summary.json"

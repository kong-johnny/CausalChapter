#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CC_BASE_MODEL:?Set CC_BASE_MODEL to your official Qwen2.5-3B-Instruct directory}"
export USE_TF=0 USE_FLAX=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python3 scripts/verify_artifacts.py --profile clean
python3 scripts/prepare_eval_features.py --archive data/avlecture/lcds_visual_features.tar.gz --destination data/lcds
export CC_VIS_CACHE_DIR="$ROOT/data/lcds/vis2d_cache/avlecture"
SEGMENT_OUTPUT="${CC_SEG_OUTPUT_DIR:-$ROOT/outputs/clean_epoch1_segmentation}"
python3 -m segment.infer_lcds --checkpoint checkpoints/lcds_longformer_reconstruction \
  --text data/avlecture/test.jsonl --visual-cache "$CC_VIS_CACHE_DIR" --output "$SEGMENT_OUTPUT"
export CC_TEST_PRED_BOUNDARY="$SEGMENT_OUTPUT/boundaries.jsonl"
export CC_CHAPTER_TIMES="$SEGMENT_OUTPUT/chapter_times.jsonl"
export CC_CHECKPOINT_DIR="$ROOT/checkpoints/qwen2p5_3b_clean_epoch1"
export CC_EVAL_SAVE_DIR="${CC_EVAL_SAVE_DIR:-$ROOT/outputs/clean_epoch1_reproduced}"
export CC_EVAL_SCOPE=pred CC_EVAL_PROTOCOL=strict CC_CAPTION_MATCHING=lexical CC_FORBID_GT_SEGMENTS=1
export CC_SEARCH_DEV_ONLY=0 CC_EVAL_LIMIT=0 CC_FUSION_MODE=text_only CC_NUM_QUERY_TOKENS=0
export CC_LORA_R=32 CC_LORA_ALPHA=64 CC_INPUT_MAX_TOKENS=8192
export CC_CAUSAL_MODE=light CC_CAUSAL_TOPK=3 CC_SEMANTIC_BACKEND=hashing CC_USE_FEWSHOT=0
export CC_RETRIEVAL_NEAR_SIZE=3 CC_RETRIEVAL_SEMANTIC_SIZE=10
export CC_RETRIEVAL_INTERVENTION_TOKENS=16 CC_RETRIEVAL_VARIANT_MICROBATCH=2
export CC_GEN_NUM_BEAMS=3 CC_GEN_REPETITION_PENALTY=1.005 CC_GEN_NO_REPEAT_NGRAM_SIZE=3 CC_SEED=42
python3 -m generate.eval_env

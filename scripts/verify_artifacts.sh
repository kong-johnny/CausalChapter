#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
sha256sum \
  checkpoints/qwen2p5_3b_crossattn_clip_best/qformer.pt \
  checkpoints/qwen2p5_3b_crossattn_clip_best/shared_lora/adapter_model.safetensors \
  boundaries/boundaries_12_thr0.35_gap6.jsonl \
  outputs/best_eval/eval_summary.json \
  outputs/best_eval/eval_details_pred.jsonl

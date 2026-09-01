#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ASR + CLIP cross-attn Q=8 joint continuation.

Warm-start from the strong text LoRA, initialize QFormer from the previous
cross-attn checkpoint when available, then train QFormer + LoRA with separate
LRs.  Intended fallback when QFormer-only underperforms text-only.
"""
import os
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    __package__ = "generate"
    _THIS_FILE = Path(__file__).resolve()
    _PROJECT_ROOT = _THIS_FILE.parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import generate.train as tr

# Optional reproducibility knob for small reruns. Backward-compatible: default remains 42.
tr.SEED = int(os.environ.get("CC_SEED", tr.SEED))

tr.FUSION_MODE = "cross_attn"
tr.VIS_CACHE_DIR = "/mnt/nas2/duanmo/data/avlecture/vis2d_cache/avlecture"
tr.VIS_DIM = 512
tr.NUM_QUERY_TOKENS = 8

tr.OUTPUT_DIR = os.environ.get(
    "CC_OUTPUT_DIR",
    "/mnt/nas2/duanmo/data/video_chapter/output_crossattn_clip_q8_textinit_joint_nf_top5",
)
tr.BASE_MODEL = "/mnt/nas2/duanmo/data/video_chapter/models/Qwen2.5-3B-Instruct"
tr.TEST_PRED_BOUNDARY = os.environ.get(
    "CC_TEST_PRED_BOUNDARY",
    "/mnt/nas2/duanmo/data/video_chapter/output_segment_predopt_endpos_z02/boundaries_15_thr0.3_gap0.jsonl",
)

tr.INIT_LORA_ADAPTER = os.environ.get(
    "CC_INIT_LORA_ADAPTER",
    "/mnt/nas2/duanmo/data/video_chapter/output_llama_ctx/checkpoint-150",
)
# If the QFormer-only run produces a checkpoint, launch with CC_INIT_QFORMER=<that dir>.
tr.INIT_QFORMER = os.environ.get(
    "CC_INIT_QFORMER",
    "/mnt/nas2/duanmo/data/video_chapter/output_crossattn_clip_q8/best",
)
tr.TRAIN_QFORMER_ONLY = False
tr.TRAIN_LORA_ONLY = False

tr.EPOCHS = int(os.environ.get("CC_EPOCHS", "2"))
tr.EVAL_EVERY = 1
tr.CAUSAL_TOPK = int(os.environ.get("CC_CAUSAL_TOPK", "5"))
tr.LORA_R = 64
tr.LORA_ALPHA = 128
tr.LR = float(os.environ.get("CC_LR", "1e-5"))
tr.QFORMER_LR = float(os.environ.get("CC_QFORMER_LR", "2e-4"))
tr.LORA_LR = float(os.environ.get("CC_LORA_LR", "5e-6"))
tr.WARMUP_RATIO = 0.0
tr.USE_FEWSHOT = os.environ.get("CC_USE_FEWSHOT", "0").lower() in {"1", "true", "yes"}
tr.FEWSHOT_NUM = int(os.environ.get("CC_FEWSHOT_NUM", "3"))
tr.INPUT_MAX_TOKENS = 4096
tr.PASS1_SEG_TOKENS = 4096
tr.PASS2_SEG_TOKENS = 4096
tr.MAX_LENGTH = 4096

if __name__ == "__main__":
    tr.main()

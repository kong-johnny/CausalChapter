#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ASR + CLIP cross-attn Q=8, warm-start text LoRA, train QFormer only.

Rationale: existing cross-attn from base underperforms the strong text-only
checkpoint.  Preserve the text generator and learn visual soft prompt tokens.
"""
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    __package__ = "generate"
    _THIS_FILE = Path(__file__).resolve()
    _PROJECT_ROOT = _THIS_FILE.parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import generate.train as tr

tr.FUSION_MODE = "cross_attn"
tr.VIS_CACHE_DIR = "/mnt/nas2/duanmo/data/avlecture/vis2d_cache/avlecture"
tr.VIS_DIM = 512
tr.NUM_QUERY_TOKENS = 8

tr.OUTPUT_DIR = "/mnt/nas2/duanmo/data/video_chapter/output_crossattn_clip_q8_textinit_qonly_nf_top5"
tr.BASE_MODEL = "/mnt/nas2/duanmo/data/video_chapter/models/Qwen2.5-3B-Instruct"
tr.TEST_PRED_BOUNDARY = "/mnt/nas2/duanmo/data/video_chapter/output_segment_predopt_endpos_z02/boundaries_15_thr0.3_gap0.jsonl"

tr.INIT_LORA_ADAPTER = "/mnt/nas2/duanmo/data/video_chapter/output_llama_ctx/checkpoint-150"
tr.TRAIN_QFORMER_ONLY = True

tr.EPOCHS = 3
tr.EVAL_EVERY = 1
tr.CAUSAL_TOPK = 5
tr.LORA_R = 64
tr.LORA_ALPHA = 128
tr.LR = 2e-4
tr.WARMUP_RATIO = 0.0
tr.USE_FEWSHOT = False
tr.FEWSHOT_NUM = 3
tr.INPUT_MAX_TOKENS = 4096
tr.PASS1_SEG_TOKENS = 4096
tr.PASS2_SEG_TOKENS = 4096
tr.MAX_LENGTH = 4096

if __name__ == "__main__":
    tr.main()

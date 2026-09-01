#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qwen2.5-3B ASR + CLIP/vis2d cross-attn experiment, Q=4."""
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
tr.NUM_QUERY_TOKENS = 4

tr.OUTPUT_DIR = "/mnt/nas2/duanmo/data/video_chapter/output_crossattn_clip_q4"
tr.BASE_MODEL = "/mnt/nas2/duanmo/data/video_chapter/models/Qwen2.5-3B-Instruct"
tr.EPOCHS = 2
tr.EVAL_EVERY = 1
tr.LORA_R = 64
tr.LORA_ALPHA = 128
tr.LR = 5e-5
tr.WARMUP_RATIO = 0.0
tr.USE_FEWSHOT = True
tr.FEWSHOT_NUM = 3

if __name__ == "__main__":
    tr.main()

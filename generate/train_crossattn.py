#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple training: 10 epochs, text_only, use project pipeline"""
# Redirect to original with modded settings
import os, sys
from pathlib import Path
if __name__ == "__main__" and __package__ is None:
    __package__ = "generate"
    _THIS_FILE = Path(__file__).resolve()
    _PROJECT_ROOT = _THIS_FILE.parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

# Override settings before import
import generate.train as tr
tr.FUSION_MODE = "cross_attn"
tr.NUM_QUERY_TOKENS = 4
tr.OUTPUT_DIR = "/mnt/nas2/duanmo/data/video_chapter/output_crossattn"
tr.BASE_MODEL = "/mnt/nas2/duanmo/data/video_chapter/models/Qwen2.5-3B-Instruct"
tr.EPOCHS = 2
tr.EVAL_EVERY = 1
tr.LORA_R = 64
tr.LORA_ALPHA = 128
tr.USE_FEWSHOT = True
tr.WARMUP_RATIO = 0.03
tr.LR = 5e-5

# Monkey-patch to add VIS tokens to Qwen tokenizer
_orig_from_pretrained = tr.AutoTokenizer.from_pretrained.__func__
def _patched_from_pretrained(cls, *args, **kwargs):
    tok = _orig_from_pretrained(cls, *args, **kwargs)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    # Add VIS tokens if not in vocab
    vis_tokens = ['<VIS_0>', '<VIS_1>', '<VIS_2>', '<VIS_3>']
    for vt in vis_tokens:
        if vt not in tok.get_vocab():
            tok.add_tokens([vt], special_tokens=True)
            print(f'[TOKENIZER] added {vt}')
    return tok
tr.AutoTokenizer.from_pretrained = classmethod(_patched_from_pretrained)

if __name__ == "__main__":
    tr.main()
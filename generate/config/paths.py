# -*- coding: utf-8 -*-
"""Centralized data paths for the CausalChapter pipeline.

All paths are relative to the project root (video_chapter/).
Data is expected under data/avlecture/ — symlink or copy the AVLecture dataset there.
"""

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "avlecture")

# ---- Ground-truth annotation files ----
TRAIN_GT_JSONL = os.path.join(DATA_DIR, "train_subheading_abstract_mixed_youhuahou.jsonl")
DEV_GT_JSONL = os.path.join(DATA_DIR, "dev_subheading_abstract_mixed_youhuahou.jsonl")
TEST_GT_JSONL = os.environ.get("CC_EVAL_GT_JSONL", os.path.join(DATA_DIR, "test_subheading_abstract.jsonl"))

TRAIN_MIX_JSONL = os.path.join(DATA_DIR, "train_subheading_abstract_mixed_youhuahou.jsonl")
DEV_MIX_JSONL = os.path.join(DATA_DIR, "dev_subheading_abstract_mixed_youhuahou.jsonl")

# ---- Utterance-level text files ----
TRAIN_TEXT_JSONL = os.path.join(DATA_DIR, "train_mixed_youhuahou.jsonl")
DEV_TEXT_JSONL = os.path.join(DATA_DIR, "dev_mixed_youhuahou.jsonl")
TEST_TEXT_JSONL = os.environ.get("CC_EVAL_TEXT_JSONL", os.path.join(DATA_DIR, "test.jsonl"))

TRAIN_MIX_TEXT_JSONL = os.path.join(DATA_DIR, "train_mixed_youhuahou.jsonl")
DEV_MIX_TEXT_JSONL = os.path.join(DATA_DIR, "dev_mixed_youhuahou.jsonl")

# ---- Boundary predictions (from segment pipeline) ----
TEST_LABEL_JSON = os.path.join(DATA_DIR, "avl_test.json")
TEST_PRED_BOUNDARY = os.path.join(DATA_DIR, "test_boundaries.jsonl")

# ---- Visual feature cache (unused in text_only mode) ----
VIS_CACHE_DIR = os.path.join(DATA_DIR, "vis2d_cache", "avlecture")

print(f"[paths] DATA_DIR={DATA_DIR}")
print(f"[paths] TRAIN_GT_JSONL={os.path.basename(TRAIN_GT_JSONL)}")
print(f"[paths] TEST_GT_JSONL={os.path.basename(TEST_GT_JSONL)}")

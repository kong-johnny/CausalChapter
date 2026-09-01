# -*- coding: utf-8 -*-
"""Evaluation data construction"""
from tqdm import tqdm

from ..utils.io_utils import load_jsonl, build_example_map
from ..data.segment_builder import build_gt_segments, segment_by_boundary, parse_stet_list, match_pred_segments_to_gt


def build_gt_eval_data(test_gt_jsonl, test_text_jsonl):
    gt_rows = load_jsonl(test_gt_jsonl)
    text_rows = load_jsonl(test_text_jsonl)

    gt_map = build_example_map(gt_rows)
    text_map = build_example_map(text_rows)

    data = []
    for eid, gt_video in tqdm(gt_map.items(), desc="Prepare GT eval"):
        if eid not in text_map:
            continue
        gt_segments = build_gt_segments(gt_video, text_map[eid])
        if len(gt_segments) == 0:
            continue

        data.append({
            "example_id": eid,
            "segments": [x["segment_text"] for x in gt_segments],
            "spans": [(x["start_idx"], x["end_idx"]) for x in gt_segments],
            "times": [(x["start_time"], x["end_time"]) for x in gt_segments],
            "gt_titles": [x["target_subheading"] for x in gt_segments],
        })
    return data


def build_pred_eval_data(test_text_jsonl, test_gt_jsonl, test_pred_boundary):
    text_rows = load_jsonl(test_text_jsonl)
    gt_rows = load_jsonl(test_gt_jsonl)
    pred_rows = load_jsonl(test_pred_boundary)

    gt_map = build_example_map(gt_rows)
    text_map = build_example_map(text_rows)

    if len(text_rows) != len(pred_rows):
        raise ValueError("test text rows != test pred boundary rows")

    data = []
    for text_video, bd in tqdm(zip(text_rows, pred_rows), total=len(text_rows), desc="Prepare Pred eval"):
        eid = text_video["example_id"]
        if eid not in gt_map:
            continue

        stet_list = parse_stet_list(text_video.get("stet", []))
        utt_texts = text_video.get("text", [])
        pred_segments = segment_by_boundary(utt_texts, stet_list, bd.get("predictions", []))
        gt_segments = build_gt_segments(gt_map[eid], text_video)

        if len(pred_segments) == 0 or len(gt_segments) == 0:
            continue

        # Use Hungarian matching to align predicted segments to GT titles
        matched = match_pred_segments_to_gt(pred_segments, gt_segments)

        if len(matched) == 0:
            continue

        data.append({
            "example_id": eid,
            "segments": [x["segment_text"] for x in matched],
            "spans": [(x["start_idx"], x["end_idx"]) for x in matched],
            "times": [(x["start_time"], x["end_time"]) for x in matched],
            "gt_titles": [x["target_subheading"] for x in matched],
            "gt_events": [
                {"start": x["start_time"], "end": x["end_time"], "sentence": x["target_subheading"]}
                for x in matched
            ],
        })
    return data
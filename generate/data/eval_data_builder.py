# -*- coding: utf-8 -*-
"""Evaluation data construction"""
from tqdm import tqdm

from ..utils.io_utils import load_jsonl, build_example_map
from ..data.segment_builder import build_gt_segments, segment_by_boundary, parse_stet_list, match_pred_segments_to_gt


def build_strict_pred_eval_data(test_text_jsonl, test_gt_jsonl, test_pred_boundary, chapter_times=None):
    """Keep every prediction; annotations never select generation inputs."""
    text_rows = load_jsonl(test_text_jsonl)
    gt_rows = load_jsonl(test_gt_jsonl)
    pred_rows = load_jsonl(test_pred_boundary)
    def index(rows):
        result = {}
        for row in rows:
            eid = row["example_id"]
            if isinstance(eid, list):
                # Legacy segment output repeats the same ID per utterance.
                if not eid or any(value != eid[0] for value in eid):
                    raise ValueError("Expected one consistent video ID")
                if len(eid) != 1 and len(eid) != len(row.get("predictions", [])):
                    raise ValueError("Repeated video IDs must match boundary length")
                eid = eid[0]
            if eid in result:
                raise ValueError(f"Duplicate video ID: {eid}")
            result[eid] = row
        return result
    text_map, gt_map, pred_map = map(index, (text_rows, gt_rows, pred_rows))
    if set(text_map) != set(gt_map) or set(text_map) != set(pred_map):
        raise ValueError("Strict evaluation requires exactly matching video IDs")
    time_map = index(load_jsonl(chapter_times)) if chapter_times else None
    if time_map is not None and set(time_map) != set(text_map):
        raise ValueError("Chapter timeline IDs must match the evaluation split")
    data = []
    for eid, text in text_map.items():
        labels = pred_map[eid]["predictions"]
        if len(labels) != len(text["text"]):
            raise ValueError(f"Boundary length mismatch: {eid}")
        if any(value not in (0, 1, "O", "B-EOP") for value in labels):
            raise ValueError(f"Invalid boundary label: {eid}")
        predicted = segment_by_boundary(text["text"], parse_stet_list(text["stet"]), labels, keep_empty=True)
        reference = build_gt_segments(gt_map[eid], text)
        if not predicted or not reference:
            raise ValueError(f"Empty strict prediction/reference video: {eid}")
        if time_map is not None:
            timeline = time_map[eid]
            expected_indices = [i for i, value in enumerate(labels[:-1]) if value in (1, "B-EOP")]
            if timeline["boundary_indices"] != expected_indices or len(timeline["chapters"]) != len(predicted):
                raise ValueError(f"Timeline/boundary mismatch: {eid}")
            previous_end = 0.0
            for segment, (start, end) in zip(predicted, timeline["chapters"]):
                if start != previous_end or end <= start:
                    raise ValueError(f"Invalid timeline: {eid}")
                segment["start_time"], segment["end_time"] = start, end
                previous_end = end
            if abs(previous_end - float(text["stet"][-1][1])) > 1e-5:
                raise ValueError(f"Incomplete timeline: {eid}")
        data.append({"example_id": eid, "evaluation_protocol": "strict",
                     "segments": [s["segment_text"] for s in predicted],
                     "spans": [(s["start_idx"], s["end_idx"]) for s in predicted],
                     "times": [(s["start_time"], s["end_time"]) for s in predicted],
                     "gt_titles": [s["target_subheading"] for s in reference],
                     "gt_events": [{"start": s["start_time"], "end": s["end_time"], "sentence": s["target_subheading"]} for s in reference]})
    return data


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

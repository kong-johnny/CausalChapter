# -*- coding: utf-8 -*-
"""Training sample construction"""
from collections import defaultdict
from tqdm import tqdm

from ..utils.io_utils import load_jsonl, build_example_map
from ..data.segment_builder import (
    build_gt_segments, segment_by_boundary,
    match_pred_segments_to_gt, parse_stet_list,
)


def load_gt_sample(gt_jsonl_path, text_jsonl_path, source="gt"):
    gt_rows = load_jsonl(gt_jsonl_path)
    text_rows = load_jsonl(text_jsonl_path)
    gt_map = build_example_map(gt_rows)
    text_map = build_example_map(text_rows)

    all_segments = []
    for eid, gt_video in tqdm(gt_map.items(), desc=f"Load GT samples ({source})"):
        if eid not in text_map:
            continue
        text_video = text_map[eid]
        segments = build_gt_segments(gt_video, text_video)
        for seg in segments:
            seg["source"] = source
        all_segments.extend(segments)

    return all_segments


def load_pred_sample(pred_txt_path, text_jsonl_path, source="pred"):
    import json
    text_map = build_example_map(load_jsonl(text_jsonl_path))

    all_segments = []
    with open(pred_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            pred_data = json.loads(line.strip())
            eid = pred_data["example_id"]
            if eid not in text_map:
                continue
            text_video = text_map[eid]
            segments = segment_by_boundary(
                text_video["text"],
                parse_stet_list(text_video.get("stet", [])),
                pred_data["predictions"]
            )
            for seg in segments:
                seg["example_id"] = eid
                seg["source"] = source
            all_segments.extend(segments)
    return all_segments


def build_training_samples(
    train_mode,
    train_gt_jsonl,
    dev_gt_jsonl,
    train_text_jsonl,
    dev_text_jsonl,
    train_pred_boundary=None,
    dev_pred_boundary=None,
    alpha_time=0.35,
    alpha_utt=0.35,
    alpha_text=0.30,
    min_match_score=0.0,
):
    """Historical builder: combines train AND dev; not a clean-holdout API.

    Retained for auditing old experiments only. New experiments must use
    build_train_only_samples and an independent predicted-boundary dev scorer.
    """
    assert train_mode in ["gt_only", "gt_plus_pred_raw", "gt_plus_pred_opt"]

    train_gt_rows = load_jsonl(train_gt_jsonl)
    dev_gt_rows = load_jsonl(dev_gt_jsonl)
    train_text_rows = load_jsonl(train_text_jsonl)
    dev_text_rows = load_jsonl(dev_text_jsonl)

    train_gt_map = build_example_map(train_gt_rows)
    dev_gt_map = build_example_map(dev_gt_rows)
    train_text_map = build_example_map(train_text_rows)
    dev_text_map = build_example_map(dev_text_rows)

    samples = []

    # GT portion
    for eid, gt_video in tqdm(train_gt_map.items(), desc="Build GT train samples"):
        if eid not in train_text_map:
            continue
        gt_segments = build_gt_segments(gt_video, train_text_map[eid])
        samples.extend(gt_segments)

    for eid, gt_video in tqdm(dev_gt_map.items(), desc="Build GT dev samples"):
        if eid not in dev_text_map:
            continue
        gt_segments = build_gt_segments(gt_video, dev_text_map[eid])
        samples.extend(gt_segments)

    # Pred portion
    if train_mode != "gt_only":
        if train_pred_boundary is None or dev_pred_boundary is None:
            raise ValueError(f"{train_mode} requires --train_pred_boundary and --dev_pred_boundary")

        train_pred_rows = load_jsonl(train_pred_boundary)
        dev_pred_rows = load_jsonl(dev_pred_boundary)

        if len(train_pred_rows) != len(train_text_rows):
            raise ValueError("train_pred_boundary length != train_text_jsonl length")
        if len(dev_pred_rows) != len(dev_text_rows):
            raise ValueError("dev_pred_boundary length != dev_text_jsonl length")

        # train pred
        for text_video, bd in tqdm(zip(train_text_rows, train_pred_rows), total=len(train_text_rows)):
            eid = text_video["example_id"]
            if eid not in train_gt_map:
                continue
            utt_texts = text_video.get("text", [])
            stet_list = parse_stet_list(text_video.get("stet", []))

            gt_segments = build_gt_segments(train_gt_map[eid], text_video)
            pred_segments = segment_by_boundary(utt_texts, stet_list, bd.get("predictions", []))
            pred_segments = [{**x, "example_id": eid, "source": "pred"} for x in pred_segments]

            matched = match_pred_segments_to_gt(
                pred_segments, gt_segments,
                alpha_time=alpha_time, alpha_utt=alpha_utt,
                alpha_text=alpha_text, min_score=min_match_score
            )

            samples.extend(pred_segments)
            samples.extend(matched)

        # dev pred
        for text_video, bd in tqdm(zip(dev_text_rows, dev_pred_rows), total=len(dev_text_rows)):
            eid = text_video["example_id"]
            if eid not in dev_gt_map:
                continue
            utt_texts = text_video.get("text", [])
            stet_list = parse_stet_list(text_video.get("stet", []))
            gt_segments = build_gt_segments(dev_gt_map[eid], text_video)
            pred_segments = segment_by_boundary(utt_texts, stet_list, bd.get("predictions", []))
            pred_segments = [{**x, "example_id": eid} for x in pred_segments]

            matched = match_pred_segments_to_gt(
                pred_segments, gt_segments,
                alpha_time=alpha_time, alpha_utt=alpha_utt,
                alpha_text=alpha_text, min_score=min_match_score
            )
            samples.extend(matched)

    src_count = defaultdict(int)
    for s in samples:
        src_count[s["source"]] += 1

    print(f"[DATA] total training samples: {len(samples)}")
    print(f"[DATA] source distribution: {dict(src_count)}")

    return samples


def build_train_only_samples(train_gt_jsonl, train_text_jsonl, dev_text_jsonl):
    """Construct gradient samples exclusively from audited train rows."""
    from .split_audit import audited_train_rows
    texts, annotations, manifest = audited_train_rows(
        train_text_jsonl, train_gt_jsonl, dev_text_jsonl)
    text_map = build_example_map(texts)
    samples = []
    for row in annotations:
        eid = row["example_id"]
        if eid not in text_map:
            raise ValueError("Exact training annotation/transcript ID mismatch")
        samples.extend(build_gt_segments(row, text_map[eid]))
    train_ids = {row["example_id"] for row in texts}
    if any(sample["example_id"] not in train_ids for sample in samples):
        raise ValueError("Non-train example entered gradient samples")
    return samples, manifest

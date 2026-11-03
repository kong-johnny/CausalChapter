# -*- coding: utf-8 -*-
"""GT / Pred segment construction and matching"""
from collections import defaultdict
import numpy as np

from ..utils.text_utils import normalize_text, text_jaccard, safe_float
from ..utils.match_utils import temporal_iou, span_iou, hungarian_match


def parse_stet_list(stet):
    out = []
    for x in stet:
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            s = safe_float(x[0])
            e = safe_float(x[1])
        else:
            s, e = 0.0, 0.0
        out.append((s, e))
    return out


def time_to_utt_span(seg_start, seg_end, stet_list):
    overlaps = []
    for i, (u_s, u_e) in enumerate(stet_list):
        inter = max(0.0, min(seg_end, u_e) - max(seg_start, u_s))
        if inter > 0:
            overlaps.append(i)

    if len(overlaps) > 0:
        return overlaps[0], overlaps[-1] + 1

    center = (seg_start + seg_end) / 2.0
    best_i = 0
    best_dist = 1e18
    for i, (u_s, u_e) in enumerate(stet_list):
        uc = (u_s + u_e) / 2.0
        d = abs(center - uc)
        if d < best_dist:
            best_dist = d
            best_i = i
    return best_i, best_i + 1


def build_gt_segments(gt_video, text_video):
    example_id = gt_video["example_id"]
    stet_list = parse_stet_list(text_video.get("stet", []))
    gt_items = gt_video.get("subheading_abstract", [])

    segments = []
    for seg in gt_items:
        seg_text = normalize_text(seg.get("text", ""))
        subheading = normalize_text(seg.get("subheading", ""))
        seg_start = safe_float(seg.get("start", 0))
        seg_end = safe_float(seg.get("end", 0))

        if not seg_text or not subheading:
            continue

        if len(stet_list) > 0:
            start_idx, end_idx = time_to_utt_span(seg_start, seg_end, stet_list)
        else:
            idx = int(seg.get("index", len(segments)))
            start_idx, end_idx = idx, idx + 1

        segments.append({
            "example_id": example_id,
            "segment_text": seg_text,
            "target_subheading": subheading,
            "start_time": seg_start,
            "end_time": seg_end,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "source": "gt",
            "gt_index": int(seg.get("index", len(segments))),
        })
    return segments


def segment_by_boundary(utt_texts, stet_list, predictions, keep_empty=False):
    n = len(utt_texts)
    if n == 0:
        return []

    cut_points = [0]
    for i, pred in enumerate(predictions[:n]):
        if pred == "B-EOP" or pred == 1:
            cut_points.append(i + 1)

    if cut_points[-1] < n:
        cut_points.append(n)

    segments = []
    for k in range(len(cut_points) - 1):
        s = cut_points[k]
        e = cut_points[k + 1]
        if s >= e:
            continue
        seg_text = " ".join([normalize_text(x) for x in utt_texts[s:e]]).strip()
        if len(seg_text) == 0 and not keep_empty:
            continue
        start_time = stet_list[s][0] if len(stet_list) > s else float(s)
        end_time = stet_list[e - 1][1] if len(stet_list) >= e else float(e)
        segments.append({
            "segment_text": seg_text,
            "start_idx": s,
            "end_idx": e,
            "start_time": start_time,
            "end_time": end_time,
        })

    if len(segments) == 0:
        seg_text = " ".join([normalize_text(x) for x in utt_texts]).strip()
        end_time = stet_list[-1][1] if len(stet_list) > 0 else float(n)
        segments.append({
            "segment_text": seg_text,
            "start_idx": 0,
            "end_idx": n,
            "start_time": 0.0,
            "end_time": end_time,
        })

    return segments


def match_pred_segments_to_gt(
    pred_segments,
    gt_segments,
    alpha_time=0.35,
    alpha_utt=0.35,
    alpha_text=0.30,
    min_score=0.0
):
    if len(pred_segments) == 0 or len(gt_segments) == 0:
        return []

    score_matrix = np.zeros((len(pred_segments), len(gt_segments)), dtype=np.float32)
    for i, p in enumerate(pred_segments):
        for j, g in enumerate(gt_segments):
            s_time = temporal_iou(p["start_time"], p["end_time"], g["start_time"], g["end_time"])
            s_utt = span_iou(p["start_idx"], p["end_idx"], g["start_idx"], g["end_idx"])
            s_txt = text_jaccard(p["segment_text"], g["segment_text"])
            score = alpha_time * s_time + alpha_utt * s_utt + alpha_text * s_txt
            score_matrix[i, j] = score

    pairs = hungarian_match(score_matrix)

    matched = []
    for pi, gj in pairs:
        score = float(score_matrix[pi, gj])
        if score < min_score:
            continue
        p = pred_segments[pi]
        g = gt_segments[gj]
        matched.append({
            "example_id": g["example_id"],
            "segment_text": p["segment_text"],
            "target_subheading": g["target_subheading"],
            "start_time": p["start_time"],
            "end_time": p["end_time"],
            "start_idx": p["start_idx"],
            "end_idx": p["end_idx"],
            "source": "pred",
            "matched_gt_index": g["gt_index"],
            "match_score": score,
        })
    return matched


def group_samples_by_video(samples):
    by_eid = defaultdict(list)
    for s in samples:
        by_eid[s["example_id"]].append(s)
    for eid in by_eid:
        by_eid[eid].sort(key=lambda x: (x["start_idx"], x["end_idx"]))
    return by_eid

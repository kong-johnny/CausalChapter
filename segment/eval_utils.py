# -*- coding: utf-8 -*-
"""
Evaluation utilities for video topic segmentation.

Metrics:
  - F1: Exact boundary match F1 score
  - F1@k: F1 with k-sentence tolerance window (k=30 default)
  - BS@k: Boundary Similarity at window k
  - tIoU: Temporal Intersection-over-Union of predicted vs GT segments
"""
import numpy as np
from typing import List


def compute_boundary_similarity(predictions, references, k=30):
    """Boundary Similarity (BS) at window size k.

    BS@k measures how well predicted boundaries match GT boundaries
    within a tolerance window of k sentences.

    Args:
        predictions: list of lists of int labels (0/1 or 0/B-EOP)
        references: list of lists of int labels
        k: tolerance window size

    Returns:
        float: BS@k score
    """
    try:
        from segeval.window.windowdiff import window_diff as WD

        wd_scores = []
        for pred, ref in zip(predictions, references):
            pred_int = [1 if p in (1, "B-EOP") else 0 for p in pred]
            ref_int = [1 if r in (1, "B-EOP") else 0 for r in ref]
            if len(pred_int) != len(ref_int):
                continue
            if sum(pred_int) == 0 and sum(ref_int) == 0:
                wd_scores.append(1.0)
            elif sum(pred_int) == 0 or sum(ref_int) == 0:
                wd_scores.append(0.0)
            else:
                wd = WD(pred_int, ref_int, window_size=k)
                wd_scores.append(1.0 - wd)
        return float(np.mean(wd_scores)) if wd_scores else 0.0
    except (ImportError, Exception):
        return _bs_fallback(predictions, references, k)


def _bs_fallback(predictions, references, k=30):
    """Fallback BS@k implementation without segeval."""
    scores = []
    for pred, ref in zip(predictions, references):
        p = [1 if x in (1, "B-EOP") else 0 for x in pred]
        r = [1 if x in (1, "B-EOP") else 0 for x in ref]
        if len(p) != len(r) or len(p) == 0:
            continue
        n = len(p)
        matches = 0
        total = 0
        for i in range(n - k):
            p_win = sum(p[i:i + k])
            r_win = sum(r[i:i + k])
            matches += 1 - abs(p_win - r_win) / k
            total += 1
        scores.append(matches / max(total, 1))
    return float(np.mean(scores)) if scores else 0.0


def compute_f1_at_k(predictions, references, k=30):
    """F1 score with k-sentence tolerance window.

    A predicted boundary is considered correct if there is a GT boundary
    within k sentences before or after it.

    Args:
        predictions: list of lists of boundary predictions
        references: list of lists of GT boundary labels
        k: tolerance window in number of sentences

    Returns:
        float: F1@k score
    """
    total_tp, total_fp, total_fn = 0, 0, 0

    for pred, ref in zip(predictions, references):
        pred_idx = {i for i, x in enumerate(pred) if x in (1, "B-EOP")}
        ref_idx = {i for i, x in enumerate(ref) if x in (1, "B-EOP")}

        tp = 0
        matched_ref = set()
        for pi in sorted(pred_idx):
            for ri in sorted(ref_idx - matched_ref):
                if abs(pi - ri) <= k:
                    tp += 1
                    matched_ref.add(ri)
                    break

        fp = len(pred_idx) - tp
        fn = len(ref_idx) - len(matched_ref)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return float(f1)


def compute_f1(predictions, references):
    """Exact boundary match F1 (k=0).

    Args:
        predictions: list of lists of boundary predictions
        references: list of lists of GT boundary labels

    Returns:
        float: F1 score
    """
    from sklearn.metrics import f1_score as sk_f1

    all_preds = []
    all_refs = []
    for pred, ref in zip(predictions, references):
        p = [1 if x in (1, "B-EOP") else 0 for x in pred]
        r = [1 if x in (1, "B-EOP") else 0 for x in ref]
        all_preds.extend(p)
        all_refs.extend(r)

    if not all_preds:
        return 0.0
    return float(sk_f1(all_refs, all_preds))


def compute_tiou(predictions, references, stet_lists):
    """Temporal IoU: average IoU of predicted vs GT segments.

    Args:
        predictions: list of lists of boundary predictions
        references: list of lists of GT boundary labels
        stet_lists: list of lists of [start_time, end_time] for each utterance

    Returns:
        float: tIoU score
    """
    tious = []
    for pred, ref, stet in zip(predictions, references, stet_lists):
        pred_idx = {i for i, x in enumerate(pred) if x in (1, "B-EOP")}
        ref_idx = {i for i, x in enumerate(ref) if x in (1, "B-EOP")}

        if not pred_idx or not ref_idx:
            continue

        # Build segments for both pred and GT
        def build_segments(boundaries, stet_list):
            segments = []
            start = 0
            for b in sorted(boundaries):
                if b < len(stet_list):
                    seg_end = stet_list[b][1]
                    seg_start_val = stet_list[start][0] if start < len(stet_list) else 0
                    segments.append((seg_start_val, seg_end))
                    start = b + 1
            # Last segment
            if start < len(stet_list):
                segments.append((stet_list[start][0], stet_list[-1][1]))
            return segments

        pred_segs = build_segments(pred_idx, stet)
        ref_segs = build_segments(ref_idx, stet)

        # Compute best matching IoU
        if not pred_segs or not ref_segs:
            continue

        total_iou = 0.0
        matches = 0
        used_ref = set()
        for ps, pe in pred_segs:
            best_iou = 0.0
            best_j = -1
            for j, (rs, re) in enumerate(ref_segs):
                if j in used_ref:
                    continue
                intersection = max(0.0, min(pe, re) - max(ps, rs))
                union = max(pe, re) - min(ps, rs)
                iou = intersection / max(union, 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                total_iou += best_iou
                used_ref.add(best_j)
                matches += 1

        if matches > 0:
            tious.append(total_iou / max(matches, 1))

    return float(np.mean(tious)) if tious else 0.0


def compute_all_metrics(predictions, references, stet_lists=None):
    """Compute all segmentation metrics.

    Args:
        predictions: list of lists of boundary predictions
        references: list of lists of GT boundary labels
        stet_lists: list of lists of [start, end] times (for tIoU)

    Returns:
        dict: {F1, F1@30, BS@30, tIoU}
    """
    results = {
        "F1": compute_f1(predictions, references) * 100.0,
        "F1@30": compute_f1_at_k(predictions, references, k=30) * 100.0,
        "BS@30": compute_boundary_similarity(predictions, references, k=30) * 100.0,
    }

    if stet_lists is not None:
        results["tIoU"] = compute_tiou(predictions, references, stet_lists) * 100.0
    else:
        results["tIoU"] = 0.0

    return results
# -*- coding: utf-8 -*-
"""IoU / Hungarian matching utilities"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def temporal_iou(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def span_iou(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def hungarian_match(score_matrix):
    if score_matrix.size == 0:
        return []
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    return list(zip(row_ind.tolist(), col_ind.tolist()))
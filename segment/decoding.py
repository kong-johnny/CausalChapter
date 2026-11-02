"""Deterministic boundary decoding shared by model inference and score replay."""
from __future__ import annotations

import math


def decode_probabilities(probabilities: list[float], threshold: float, min_gap: int) -> list[int]:
    """Keep probability peaks at least ``min_gap`` utterances apart.

    The video end is implicit in the generator, so the final utterance is
    excluded before suppression. Ties prefer the earliest utterance.
    """
    if not 0 <= threshold <= 1 or min_gap < 0:
        raise ValueError("threshold must be in [0, 1] and min_gap must be nonnegative")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("probabilities must be finite values in [0, 1]")
    candidates = [i for i, score in enumerate(probabilities[:-1]) if score >= threshold]
    selected: list[int] = []
    for index in sorted(candidates, key=lambda i: (-probabilities[i], i)):
        if all(abs(index - previous) >= min_gap for previous in selected):
            selected.append(index)
    labels = [0] * len(probabilities)
    for index in selected:
        labels[index] = 1
    return labels

# -*- coding: utf-8 -*-
"""Utility functions for boundary prediction format conversion."""
import json


def convert_to_boundary_format(video_predictions):
    """Convert per-utterance predictions to test_boundaries.jsonl format.

    Input format compatible with build_pred_eval_data() in the generate pipeline.

    Args:
        video_predictions: dict {example_id: {"predictions": ["O", "B-EOP", ...]}}

    Returns:
        list of dicts ready to write as JSONL
    """
    output = []
    for eid, data in video_predictions.items():
        preds = data["predictions"]
        # Duplicate example_id for each BOS position (legacy format)
        n = len(preds)
        output.append({
            "example_id": [eid] * n if n > 0 else [eid],
            "predictions": preds,
        })
    return output


def save_boundaries(video_predictions, output_path):
    """Save boundary predictions to JSONL file.

    Args:
        video_predictions: dict {example_id: {"predictions": [...]}}
        output_path: path to output JSONL file
    """
    output = convert_to_boundary_format(video_predictions)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[SAVE] {len(output)} videos -> {output_path}")


def int_to_label(preds):
    """Convert integer predictions to string labels."""
    label_map = {0: "O", 1: "B-EOP"}
    return [label_map.get(int(p), "O") for p in preds]
#!/usr/bin/env python3
"""Evaluate an existing boundary JSONL without loading model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_loader_endpos import build_boundary_labels, load_jsonl, load_subheading_and_labels
from .eval_utils import compute_all_metrics
from .time_metrics import evaluate_internal_boundaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundaries", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    text = {item["example_id"]: item for item in load_jsonl(args.text)}
    labels = build_boundary_labels(text, load_subheading_and_labels(args.ground_truth))
    raw = load_jsonl(args.boundaries)
    predicted = {}
    for item in raw:
        eid_value = item["example_id"]
        eid = eid_value[0] if isinstance(eid_value, list) else eid_value
        if eid in predicted:
            raise ValueError(f"Duplicate prediction video ID: {eid}")
        if any(value not in (0, 1, "0", "1", "O", "B-EOP") for value in item["predictions"]):
            raise ValueError(f"Unrecognized boundary label for {eid}")
        predicted[eid] = [1 if value in (1, "1", "B-EOP") else 0 for value in item["predictions"]]
    if set(predicted) != set(text) or set(labels) != set(text):
        raise ValueError("Prediction, text, and ground-truth video IDs must match exactly")
    eids = list(text)
    for eid in eids:
        if len(predicted[eid]) != len(labels[eid]) or len(predicted[eid]) != len(text[eid]["stet"]):
            raise ValueError(f"Boundary length mismatch for {eid}")
    metrics = compute_all_metrics(
        [predicted[eid] for eid in eids],
        [labels[eid] for eid in eids],
        [text[eid].get("stet", []) for eid in eids],
    )
    # Raw processed labels preserve their explicit alignment and annotated times;
    # report those separately from the historical evaluator's rebuilt labels.
    time_rows = [text[eid] for eid in eids]
    result = {"videos": len(eids), "legacy_metrics": metrics,
              "time_domain_metrics": evaluate_internal_boundaries(time_rows, predicted)}
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

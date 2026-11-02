#!/usr/bin/env python3
"""Tune decoding on development scores only; never reads the test split."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segment.decoding import decode_probabilities
from segment.time_metrics import evaluate_internal_boundaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--text", default="data/avlecture/dev.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    text = Path(args.text)
    if text.name != "dev.jsonl":
        raise ValueError("This selection tool accepts dev.jsonl only, not test data")
    rows = [json.loads(line) for line in text.read_text().splitlines() if line.strip()]
    scores_path = Path(args.scores)
    scores = json.loads(scores_path.read_text())
    candidates = []
    for anchor in ("start", "middle", "end"):
        for threshold_index in range(1, 20):
            threshold = threshold_index / 20
            for gap in (0, 2, 4, 6, 8, 10, 12):
                predictions = {eid: decode_probabilities(values, threshold, gap) for eid, values in scores.items()}
                metrics = evaluate_internal_boundaries(rows, predictions, anchor)
                candidates.append({"decoding": {"threshold": threshold, "min_gap": gap, "boundary_anchor": anchor}, "metrics": metrics})
    def rank(candidate):
        m = candidate["metrics"]
        return (m["F1@30s"], m["internal_boundary_F1"], -m["boundary_count_MAE"], -(m["matched_MAE@30s"] or 0))
    selected = max(candidates, key=rank)
    output = {
        "selection_split": "dev",
        "selection_objective": "F1@30s, then exact internal F1, then lower count MAE, then lower matched time MAE",
        "dev_sha256": hashlib.sha256(text.read_bytes()).hexdigest(),
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "candidate_count": len(candidates),
        "selected": selected,
        "best_per_anchor": {anchor: max((c for c in candidates if c["decoding"]["boundary_anchor"] == anchor), key=rank) for anchor in ("start", "middle", "end")},
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

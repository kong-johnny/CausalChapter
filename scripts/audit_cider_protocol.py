#!/usr/bin/env python3
"""Re-score fixed saved captions to isolate alignment protocol differences.

This is diagnostic. It does not select a model, regenerate captions, or choose
the highest scoring protocol as a replacement for a declared benchmark.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from pycocoevalcap.cider.cider import Cider

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate.utils.match_utils import hungarian_match
from generate.utils.text_utils import text_jaccard


def score(rows, mode):
    predictions, references = {}, {}
    pairs = 0
    for row in rows:
        generated, titles = row["predictions"], row["gt_titles"]
        if mode == "recorded":
            matches = row["matches"]
        else:
            matrix = np.zeros((len(generated), len(titles)), dtype=np.float32)
            for i, prediction in enumerate(generated):
                for j, title in enumerate(titles):
                    if mode == "lexical":
                        matrix[i, j] = text_jaccard(prediction, title)
                    else:
                        start, end = row["times"][i]
                        event = row["gt_events"][j]
                        intersection = max(0, min(end, event["end"]) - max(start, event["start"]))
                        matrix[i, j] = intersection / max(max(end, event["end"]) - min(start, event["start"]), 1e-8)
            matches = hungarian_match(matrix)
            if mode == "temporal_positive":
                matches = [(i, j) for i, j in matches if matrix[i, j] > 0]
        for i, j in matches:
            key = f"{row['example_id']}_{i}"
            predictions[key] = [generated[i]]
            references[key] = [titles[j]]
            pairs += 1
    value, _ = Cider().compute_score(references, predictions)
    return {"CIDEr": value * 100, "caption_pairs": pairs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.details).read_text().splitlines() if line.strip()]
    modes = ["recorded", "lexical"]
    if all(row.get("gt_events") for row in rows):
        modes += ["temporal_positive", "temporal_unfiltered"]
    result = {"diagnostic_only": True, "fixed_outputs": str(args.details),
              "videos": len(rows), "generated_titles": sum(len(row["predictions"]) for row in rows),
              "reference_titles": sum(len(row["gt_titles"]) for row in rows),
              "alignment_profiles": {mode: score(rows, mode) for mode in modes}}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

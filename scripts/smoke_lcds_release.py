#!/usr/bin/env python3
"""Run one dev video without annotations through the packaged inference CLI."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from segment.decoding import decode_probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/lcds_longformer_reconstruction")
    parser.add_argument("--visual-cache", default="data/lcds/vis2d_cache/avlecture")
    parser.add_argument("--output", default="outputs/lcds_release_smoke")
    args = parser.parse_args()
    row = json.loads((ROOT / "data/avlecture/dev.jsonl").read_text().splitlines()[0])
    # Only these three fields cross the inference boundary.
    unannotated = {key: row[key] for key in ("example_id", "text", "stet")}
    with tempfile.TemporaryDirectory(prefix="lcds-release-smoke-") as directory:
        text = Path(directory) / "unannotated.jsonl"
        text.write_text(json.dumps(unannotated) + "\n")
        subprocess.run([sys.executable, "-u", "-m", "segment.infer_lcds", "--checkpoint", args.checkpoint,
                        "--text", str(text), "--visual-cache", args.visual_cache, "--output", args.output],
                       cwd=ROOT, check=True)
    scores = json.loads((Path(args.output) / "scores.json").read_text())[row["example_id"]]
    reference = json.loads((ROOT / "outputs/lcds_reconstruction/dev_scores.json").read_text())[row["example_id"]]
    config = json.loads((Path(args.checkpoint) / "dev_decoding_tuning.json").read_text())["selected"]["decoding"]
    if len(scores) != len(reference):
        raise ValueError("Inference utterance count differs")
    actual = decode_probabilities(scores, config["threshold"], config["min_gap"])
    expected = decode_probabilities(reference, config["threshold"], config["min_gap"])
    if actual != expected:
        raise ValueError("Packaged inference boundaries differ from frozen dev prediction")
    report = {"status": "passed", "video": row["example_id"], "annotation_fields_passed": [],
              "max_probability_difference": max(abs(a - b) for a, b in zip(scores, reference)),
              "boundaries_identical": True}
    (Path(args.output) / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

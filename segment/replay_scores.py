"""Replay a saved probability artifact using only the Python standard library."""
import argparse
import hashlib
import json
from pathlib import Path

from .decoding import decode_probabilities
from .utils import save_boundaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--min-gap", type=int, default=6)
    args = parser.parse_args()
    predictions = {}
    with Path(args.scores).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            eid = item["example_id"]
            if eid in predictions:
                raise ValueError(f"Duplicate video: {eid}")
            labels = decode_probabilities(item["probs"], args.threshold, args.min_gap)
            predictions[eid] = {"predictions": ["B-EOP" if value else "O" for value in labels]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_boundaries(predictions, args.output)
    content = Path(args.output).read_bytes()
    print(f"videos={len(predictions)}, sha256={hashlib.sha256(content).hexdigest()}")
    if args.expected and content != Path(args.expected).read_bytes():
        raise SystemExit("ERROR: output differs from the expected boundary artifact")


if __name__ == "__main__":
    main()

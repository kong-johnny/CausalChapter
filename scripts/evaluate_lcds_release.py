#!/usr/bin/env python3
"""Validate frozen dev decoding before an explicitly requested final test."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def validate_intervals(path):
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        chapters = row["chapters"]
        if not chapters or chapters[0][0] != 0:
            raise ValueError(f"Invalid chapter coverage: {row['example_id']}")
        for index, (start, end) in enumerate(chapters):
            if end <= start or (index and chapters[index - 1][1] != start):
                raise ValueError(f"Noncontiguous/nonpositive interval: {row['example_id']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--visual-cache", required=True)
    parser.add_argument("--decoding-config", required=True)
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    config = Path(args.decoding_config)
    selection = json.loads(config.read_text())
    command = [sys.executable, "-u", "-m", "segment.train_lcds", "evaluate",
               "--output", str(output), "--data-dir", args.data_dir,
               "--visual-cache", args.visual_cache, "--decoding-config", str(config)]
    subprocess.run([*command, "--split", "dev"], check=True)
    actual = json.loads((output / "dev_metrics.json").read_text())
    expected = selection["selected"]["metrics"]
    for key, value in expected.items():
        if isinstance(value, (int, float)):
            if abs(actual[key] - value) > 1e-5:
                raise ValueError(f"Frozen dev reproduction failed: {key}: {actual[key]} != {value}")
        elif actual[key] != value:
            raise ValueError(f"Frozen dev metadata mismatch: {key}")
    validate_intervals(output / "dev_chapter_times.jsonl")
    record = {"status": "dev validated; configuration frozen before test",
              "checkpoint_sha256": digest(output / "best.pt"),
              "decoding_sha256": digest(config),
              "dev_metrics": actual}
    if args.final_test:
        marker = output / "final_test_frozen.json"
        # Never silently overwrite the audit trail or repeat a final selection run.
        with marker.open("x") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
        subprocess.run([*command, "--split", "test"], check=True)
        validate_intervals(output / "test_chapter_times.jsonl")
        print("FINAL_TEST_COMPLETE; no parameters changed after dev validation", flush=True)
    else:
        print("DEV_VALIDATED; test was not loaded", flush=True)


if __name__ == "__main__":
    main()

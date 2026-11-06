#!/usr/bin/env python3
"""Portable, streaming artifact verification; never accepts LFS pointers."""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="MANIFEST.sha256")
    parser.add_argument("--profile", choices=["all", "lcds", "clean"], default="clean")
    args = parser.parse_args()
    failures = []
    checked = 0
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected, name = line.split(maxsplit=1)
        path = Path(name.strip())
        if args.profile == "clean" and not (
            str(path).startswith("checkpoints/lcds_longformer_reconstruction/")
            or str(path).startswith("checkpoints/qwen2p5_3b_clean_epoch1/")
            or str(path).startswith("data/avlecture/") and "eval_visual_features" not in str(path)
        ):
            continue
        if args.profile == "lcds" and not (str(path).startswith("checkpoints/lcds_longformer_reconstruction/")
                                            or str(path).startswith("data/avlecture/lcds_visual_features.tar.gz")):
            continue
        checked += 1
        if not path.is_file():
            failures.append(f"MISSING: {path}; obtain the Drive weight bundle or Git LFS features as documented")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            first = stream.read(1024)
            if first.startswith(b"version https://git-lfs.github.com/spec/"):
                failures.append(f"LFS POINTER: {path}; install Git LFS and run git lfs pull")
                continue
            digest.update(first)
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected:
            failures.append(f"CHECKSUM MISMATCH: {path}")
        else:
            print(f"OK: {path}")
    if not checked:
        failures.append("No artifacts matched the selected profile")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()

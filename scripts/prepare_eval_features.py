#!/usr/bin/env python3
"""Verify and safely extract the exact released evaluation visual cache."""
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path


def main():
    data = Path(__file__).resolve().parents[1] / "data" / "avlecture"
    archive_path = data / "eval_visual_features.tar.gz"
    manifest = json.loads(archive_path.with_suffix(".gz.files.json").read_text())
    seen = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for entry in archive:
            name = entry.name
            if not entry.isfile() or name not in manifest:
                raise ValueError(f"Unexpected archive member: {name}")
            target = (data / name).resolve()
            if not target.is_relative_to(data.resolve()):
                raise ValueError(f"Unsafe archive member: {name}")
            content = archive.extractfile(entry).read()
            if hashlib.sha256(content).hexdigest() != manifest[name]:
                raise ValueError(f"Corrupted archived feature: {name}")
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != manifest[name]:
                    raise ValueError(f"Existing feature differs from release: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                    stream.write(content)
                    temporary = stream.name
                os.replace(temporary, target)
            seen.add(name)
    if seen != set(manifest):
        raise ValueError("The archive is missing expected feature files")
    actual = {path.relative_to(data).as_posix() for path in (data / "vis2d_cache" / "avlecture").glob("*.npy")}
    if actual != set(manifest):
        raise ValueError("The cache contains additional features. Use a fresh clone for the historical release profile.")
    print(f"Verified and prepared {len(seen)} visual feature files")


if __name__ == "__main__":
    main()

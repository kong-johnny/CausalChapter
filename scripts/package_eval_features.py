#!/usr/bin/env python3
"""Create a checksummed cache archive containing only evaluation videos."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    video_keys = set()
    for line in Path(args.text).read_text(encoding="utf-8").splitlines():
        eid = json.loads(line)["example_id"]
        video_keys.update([eid, eid.replace("@@", "_"), eid.replace("@@", "--"), eid.split("@@")[-1]])
    files = sorted(path for path in Path(args.cache).glob("*.npy") if path.stem.rsplit("-", 1)[0] in video_keys)
    if not files:
        raise ValueError("No evaluation features found")
    manifest = {}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=1) as archive:
        for index, path in enumerate(files):
            content = path.read_bytes()
            name = f"vis2d_cache/avlecture/{path.name}"
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
            manifest[name] = hashlib.sha256(content).hexdigest()
            if index % 1000 == 0:
                print(f"packed {index}/{len(files)}", flush=True)
    output.with_suffix(output.suffix + ".files.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps({"files": len(files), "bytes": output.stat().st_size, "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()

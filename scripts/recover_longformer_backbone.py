#!/usr/bin/env python3
"""Export a standard HF Longformer from a recovered, finetuned baseline.

This is explicitly a trained initialization, not the untouched allenai base
model. No classifier or causal weights are copied into the new LCDS head.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file
from transformers import LongformerConfig, LongformerModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    state = load_file(str(checkpoint), device="cpu")
    prefix = "text_encoder.text_encoder."
    encoder_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not encoder_state:
        raise ValueError(f"No Longformer weights with prefix {prefix}")
    config = LongformerConfig.from_dict(json.loads(Path(args.config).read_text()))
    config.architectures = ["LongformerModel"]
    model = LongformerModel(config)
    model.load_state_dict(encoder_state, strict=True)
    output = Path(args.output)
    model.save_pretrained(output, safe_serialization=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        source = Path(args.tokenizer) / name
        if source.is_file():
            shutil.copy2(source, output / name)
    provenance = {
        "initialization": "Historical supervised Longformer baseline; not the untouched allenai model",
        "source_weight_sha256": hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest(),
        "state_prefix": prefix,
        "tensor_count": len(encoder_state),
    }
    (output / "initialization_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()

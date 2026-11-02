#!/usr/bin/env python3
"""Check one real training video, including a full encoder backward pass."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from segment.train_lcds import LongformerLCDS, VideoInputs, read_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--visual-cache", required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    row = read_rows(args.data)[0]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    inputs = VideoInputs([row], tokenizer, args.visual_cache, 4096)
    print(f"input ready: {row['example_id']}, utterances={len(row['text'])}", flush=True)
    model = LongformerLCDS(args.base_model, 256, 5).to(device).train()
    print(f"model ready on {device}", flush=True)
    output = model(inputs.prepared[0], device)
    target = torch.tensor(row["labels"][:-1], device=device)
    loss = F.cross_entropy(output["logits"][:-1], target) + output["reconstruction_loss"]
    print(f"forward loss={float(loss.detach())}", flush=True)
    loss.backward()
    classifier_gradient = model.head.classifier.weight.grad
    encoder_gradient = model.encoder.get_input_embeddings().weight.grad
    assert classifier_gradient is not None and bool(torch.isfinite(classifier_gradient).all())
    assert encoder_gradient is not None and bool(torch.isfinite(encoder_gradient).all())
    print(json.dumps({"status": "passed", "loss": float(loss.detach()), "encoder_backward": True}), flush=True)


if __name__ == "__main__":
    main()

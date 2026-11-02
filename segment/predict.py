#!/usr/bin/env python3
"""Reproducible CausalChapter boundary inference and evaluation."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from .data_loader_endpos import BoundaryDataset, build_boundary_labels, load_jsonl, load_subheading_and_labels
from .decoding import decode_probabilities
from .eval_utils import compute_all_metrics
from .model import QwenBoundaryClassifier
from .utils import int_to_label, save_boundaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="models/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", default="checkpoints/segment_qwen2p5_3b_endpos_z035")
    parser.add_argument("--text", default="data/avlecture/test.jsonl")
    parser.add_argument("--ground-truth", default="data/avlecture/test_subheading_abstract.jsonl")
    parser.add_argument("--output", default="outputs/segment_reproduce/test_boundaries.jsonl")
    parser.add_argument("--metrics-output", default="outputs/segment_reproduce/metrics.json")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--min-gap", type=int, default=6)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    config = json.loads((Path(args.checkpoint) / "release_config.json").read_text(encoding="utf-8"))
    model = QwenBoundaryClassifier(
        base_model_name=args.base_model,
        num_labels=2,
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    )
    adapter_state = load_file(
        str(Path(args.checkpoint) / "lora_adapter" / "adapter_model.safetensors"),
        device="cpu",
    )
    loaded = set_peft_model_state_dict(model.base_model, adapter_state, adapter_name="default")
    if loaded.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter weights: {loaded.unexpected_keys}")
    state = torch.load(Path(args.checkpoint) / "classifier.pt", map_location="cpu", weights_only=True)
    model.classifier.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device


@torch.inference_mode()
def predict(model, dataset, device, threshold: float, min_gap: int):
    videos: dict[str, dict[str, list]] = {}
    for batch in tqdm(DataLoader(dataset, batch_size=1, shuffle=False), desc="segment"):
        eid = batch["eid"][0]
        start = int(batch["utt_start"].item())
        valid = int(batch["n_valid_bos"].item())
        if eid not in videos:
            count = int(batch["n_utts"].item())
            videos[eid] = {"score_sum": [0.0] * count, "score_count": [0] * count}
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            bos_positions=batch["bos_positions"].to(device),
            labels=None,
        )
        scores = torch.softmax(output["logits"][0, :valid].float(), dim=-1)[:, 1].cpu().tolist()
        for offset, score in enumerate(scores):
            index = start + offset
            videos[eid]["score_sum"][index] += float(score)
            videos[eid]["score_count"][index] += 1

    decoded = {}
    for eid, values in videos.items():
        probabilities = [total / max(count, 1) for total, count in zip(values["score_sum"], values["score_count"])]
        decoded[eid] = {"predictions": decode_probabilities(probabilities, threshold, min_gap)}
    return decoded


def evaluate(predictions, text_path: str, ground_truth_path: str):
    text_data = {item["example_id"]: item for item in load_jsonl(text_path)}
    labels = build_boundary_labels(text_data, load_subheading_and_labels(ground_truth_path))
    eids = [eid for eid in predictions if eid in labels]
    return compute_all_metrics(
        [predictions[eid]["predictions"] for eid in eids],
        [labels[eid] for eid in eids],
        [text_data[eid].get("stet", []) for eid in eids],
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    dataset = BoundaryDataset(args.text, args.ground_truth, tokenizer, args.max_seq_length, "test")
    model, device = load_model(args)
    predictions = predict(model, dataset, device, args.threshold, args.min_gap)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_boundaries({eid: {"predictions": int_to_label(value["predictions"])} for eid, value in predictions.items()}, args.output)
    metrics = evaluate(predictions, args.text, args.ground_truth)
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_output).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

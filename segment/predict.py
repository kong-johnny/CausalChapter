#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Boundary Prediction Inference

Load a trained segment model and generate boundary predictions for
the generate pipeline. Outputs test_boundaries.jsonl and computes metrics.
"""
import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from model import QwenBoundaryClassifier
from data_loader import BoundaryDataset
from eval_utils import compute_all_metrics
from utils import int_to_label, save_boundaries

# ═══════════════════════════════════════════════════════════════════
# Hardcoded configuration
# ═══════════════════════════════════════════════════════════════════
BASE_MODEL = "models/Qwen2.5-3B-Instruct"
CHECKPOINT_DIR = "output_segment/best"
DATA_DIR = "data/avlecture"
MAX_SEQ_LENGTH = 4096

LORA_R = 16
LORA_ALPHA = 32

OUTPUT_BOUNDARY_FILE = "data/avlecture/test_boundaries.jsonl"
# ═══════════════════════════════════════════════════════════════════


def load_model(checkpoint_dir):
    """Load the trained segment model from checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = QwenBoundaryClassifier(
        base_model_name=BASE_MODEL,
        num_labels=2,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
    )

    # Load LoRA adapter
    from peft import PeftModel
    lora_path = os.path.join(checkpoint_dir, "lora_adapter")
    model.base_model = PeftModel.from_pretrained(model.base_model, lora_path)

    # Load classifier head
    classifier_path = os.path.join(checkpoint_dir, "classifier.pt")
    if os.path.exists(classifier_path):
        model.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))

    model = model.to(device)
    model.eval()

    print(f"[LOAD] Model loaded from {checkpoint_dir}")
    return model, device


def get_predictions(model, dataset, tokenizer, device):
    """Run inference and collect per-utterance predictions."""
    video_preds = {}
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in tqdm(loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        bos_positions = batch["bos_positions"].to(device)
        eid = batch["eid"][0]
        utt_start = batch["utt_start"].item()
        n_valid = batch["n_valid_bos"].item()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bos_positions=bos_positions,
                labels=None,
            )
        logits = outputs["logits"][0, :n_valid]
        preds = torch.argmax(logits, dim=-1).cpu().tolist()

        if eid not in video_preds:
            n_utts = batch["n_utts"].item()
            video_preds[eid] = {
                "predictions": [0] * n_utts,
                "counts": [0] * n_utts,
            }

        for i in range(n_valid):
            utt_idx = utt_start + i
            if utt_idx < len(video_preds[eid]["predictions"]):
                video_preds[eid]["predictions"][utt_idx] += preds[i]
                video_preds[eid]["counts"][utt_idx] += 1

    # Majority vote
    for eid in video_preds:
        preds = video_preds[eid]["predictions"]
        counts = video_preds[eid]["counts"]
        video_preds[eid]["predictions"] = [
            1 if c > 0 and p > c // 2 else 0
            for p, c in zip(preds, counts)
        ]

    return video_preds


def main():
    print("=" * 60)
    print("Boundary Prediction Inference")
    print(f"checkpoint={CHECKPOINT_DIR}")
    print("=" * 60)

    # Load model
    model, device = load_model(CHECKPOINT_DIR)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_DIR, trust_remote_code=True, local_files_only=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    # Dataset
    test_text = os.path.join(DATA_DIR, "test.jsonl")
    test_gt = os.path.join(DATA_DIR, "test_subheading_abstract.jsonl")

    test_dataset = BoundaryDataset(
        test_text, test_gt, tokenizer, MAX_SEQ_LENGTH, "test",
    )

    # Predict
    print("\n[PREDICT] Running inference...")
    video_preds = get_predictions(model, test_dataset, tokenizer, device)

    # Convert to string labels
    pred_str = {
        eid: {"predictions": int_to_label(v["predictions"])}
        for eid, v in video_preds.items()
    }

    # Save boundaries for generate pipeline
    save_boundaries(pred_str, OUTPUT_BOUNDARY_FILE)
    print(f"[OUTPUT] Boundaries saved to {OUTPUT_BOUNDARY_FILE}")

    # ---- Compute metrics ----
    print("\n[METRICS] Computing segmentation metrics...")
    from data_loader import load_jsonl, load_subheading_and_labels, build_boundary_labels

    test_text_data = {}
    for item in load_jsonl(test_text):
        test_text_data[item["example_id"]] = item
    gt_raw = load_subheading_and_labels(test_gt)
    true_labels = build_boundary_labels(test_text_data, gt_raw)

    ref_str = {
        eid: {"predictions": int_to_label(true_labels.get(eid, []))}
        for eid in pred_str if eid in true_labels
    }

    pred_list = [v["predictions"] for v in pred_str.values() if v["predictions"]]
    ref_list = [v["predictions"] for v in ref_str.values() if v["predictions"]]
    stet_list = [
        test_text_data[eid].get("stet", [])
        for eid in pred_str if eid in test_text_data
    ]

    metrics = compute_all_metrics(pred_list, ref_list, stet_list)

    print("\n" + "=" * 60)
    print("SEGMENTATION RESULTS")
    print("=" * 60)
    print(f"  F1      = {metrics['F1']:.2f}")
    print(f"  F1@30   = {metrics['F1@30']:.2f}")
    print(f"  BS@30   = {metrics['BS@30']:.2f}")
    print(f"  tIoU    = {metrics['tIoU']:.2f}")

    # Target scores
    print(f"\n  Target: F1=56.72, F1@30=72.97, BS@30=63.02, tIoU=69.95")

    # Save metrics
    with open(os.path.join(CHECKPOINT_DIR, "predict_metrics.json"), "w") as f:
        json.dump({k: round(v, 2) for k, v in metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Boundary Prediction Training (Qwen2.5-3B Token Classifier)

Predicts B-EOP/O boundary labels for video topic segmentation.
Uses Qwen2.5-3B with LoRA + classification head.
"""
import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import QwenBoundaryClassifier
from data_loader_endpos import BoundaryDataset, load_jsonl
from eval_utils import compute_all_metrics
from utils import int_to_label, save_boundaries

# ═══════════════════════════════════════════════════════════════════
# Hardcoded configuration
# ═══════════════════════════════════════════════════════════════════
BASE_MODEL = "/mnt/nas2/duanmo/data/video_chapter/models/Qwen2.5-3B-Instruct"
DATA_DIR = "data/avlecture"

MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = float(os.environ.get("SEG_LR", "1e-4"))
WEIGHT_DECAY = 0.0
EPOCHS = int(os.environ.get("SEG_EPOCHS", "10"))
WARMUP_RATIO = 0.03
SEED = 42

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

WEIGHT_LABEL_ZERO = float(os.environ.get("SEG_WEIGHT_LABEL_ZERO", "0.2"))  # O-class CE weight

OUTPUT_DIR = os.environ.get("SEG_OUTPUT_DIR", "output_segment_timefix_endpos_z01")
EVAL_EVERY = 1
LOG_INTERVAL = 10
# ═══════════════════════════════════════════════════════════════════


def load_test_data(text_jsonl):
    """Load test data for evaluation (text + stet for tIoU)."""
    data_text = {}
    stet_lists = []
    eids = []
    for item in load_jsonl(text_jsonl):
        eid = item["example_id"]
        data_text[eid] = item
        eids.append(eid)
        stet_lists.append(item.get("stet", []))
    return data_text, stet_lists, eids


def get_predictions(model, dataset, tokenizer, device):
    """Run inference and collect per-utterance predictions."""
    model.eval()
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
        logits = outputs["logits"][0, :n_valid]  # [n_valid, 2]
        preds = torch.argmax(logits, dim=-1).cpu().tolist()

        # Merge overlapping window predictions
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

    # Majority vote on overlapping windows
    for eid in video_preds:
        preds = video_preds[eid]["predictions"]
        counts = video_preds[eid]["counts"]
        video_preds[eid]["predictions"] = [
            1 if c > 0 and p > c // 2 else 0
            for p, c in zip(preds, counts)
        ]

    return video_preds


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    writer = SummaryWriter(os.path.join(OUTPUT_DIR, "tensorboard"))

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        # Qwen2.5 uses <|im_start|> tokens; set a usable bos
        tokenizer.bos_token = tokenizer.eos_token

    print("=" * 60)
    print("Boundary Prediction Training")
    print(f"model={BASE_MODEL}")
    print(f"max_seq_length={MAX_SEQ_LENGTH}")
    print(f"lr={LEARNING_RATE}, epochs={EPOCHS}")
    print(f"weight_label_zero={WEIGHT_LABEL_ZERO}")
    print("=" * 60)

    # ---- Dataset ----
    train_gt = os.path.join(DATA_DIR, "train_subheading_abstract_mixed_youhuahou.jsonl")
    train_text = os.path.join(DATA_DIR, "train_mixed_youhuahou.jsonl")
    dev_gt = os.path.join(DATA_DIR, "dev_subheading_abstract_mixed_youhuahou.jsonl")
    dev_text = os.path.join(DATA_DIR, "dev_mixed_youhuahou.jsonl")
    test_gt = os.path.join(DATA_DIR, "test_subheading_abstract.jsonl")
    test_text = os.path.join(DATA_DIR, "test.jsonl")

    train_dataset = BoundaryDataset(
        train_text, train_gt, tokenizer, MAX_SEQ_LENGTH, "train",
    )
    dev_dataset = BoundaryDataset(
        dev_text, dev_gt, tokenizer, MAX_SEQ_LENGTH, "dev",
    )
    test_dataset = BoundaryDataset(
        test_text, test_gt, tokenizer, MAX_SEQ_LENGTH, "test",
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model ----
    model = QwenBoundaryClassifier(
        base_model_name=BASE_MODEL,
        num_labels=2,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    ).to(device)

    # ---- Optimizer + Scheduler ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    print(f"[SCHED] warmup={warmup_steps}/{total_steps} steps")

    # ---- Class weights ----
    class_weights = torch.tensor(
        [WEIGHT_LABEL_ZERO, 1.0 - WEIGHT_LABEL_ZERO],
        device=device, dtype=torch.bfloat16,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Test data for evaluation ----
    test_data_text, test_stet_lists, test_eids = load_test_data(test_text)

    # ---- Training ----
    best_f1 = -1.0
    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        step_cnt = 0
        accum_cnt = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bos_positions = batch["bos_positions"].to(device)
            n_valid = batch["n_valid_bos"].item()

            labels_raw = batch["labels"][0, :n_valid].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bos_positions=bos_positions,
                labels=None,
            )
            logits = outputs["logits"][0, :n_valid]
            loss = loss_fn(logits, labels_raw)

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            accum_cnt += 1

            if accum_cnt >= GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_cnt = 0
                global_step += 1

            total_loss += float(loss.item() * GRAD_ACCUM_STEPS)
            step_cnt += 1
            pbar.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}"})

            if global_step % LOG_INTERVAL == 0 and global_step > 0:
                writer.add_scalar("train/loss", loss.item() * GRAD_ACCUM_STEPS, global_step)

        # Flush remaining
        if accum_cnt > 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(1, step_cnt)
        print(f"[TRAIN] epoch={epoch+1}, avg_loss={avg_loss:.4f}")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # ---- Evaluation ----
        if (epoch + 1) % EVAL_EVERY == 0:
            try:
                # Dev predictions
                dev_preds = get_predictions(model, dev_dataset, tokenizer, device)
                dev_labels = dev_dataset.labels  # from data_loader

                dev_pred_str = {
                    eid: {"predictions": int_to_label(v["predictions"])}
                    for eid, v in dev_preds.items()
                }
                dev_ref_str = {
                    eid: {"predictions": int_to_label(dev_labels.get(eid, []))}
                    for eid in dev_preds if eid in dev_labels
                }

                dev_p = [v["predictions"] for v in dev_pred_str.values()]
                dev_r = [v["predictions"] for v in dev_ref_str.values()]
                dev_metrics = compute_all_metrics(dev_p, dev_r)
                print(f"[DEV] F1={dev_metrics['F1']:.2f}, F1@30={dev_metrics['F1@30']:.2f}, "
                      f"BS@30={dev_metrics['BS@30']:.2f}")
                writer.add_scalars("eval/dev", dev_metrics, epoch)

                # Test predictions
                test_preds = get_predictions(model, test_dataset, tokenizer, device)
                test_labels = test_dataset.labels

                test_pred_str = {
                    eid: {"predictions": int_to_label(v["predictions"])}
                    for eid, v in test_preds.items()
                }
                test_ref_str = {
                    eid: {"predictions": int_to_label(test_labels.get(eid, []))}
                    for eid in test_preds if eid in test_labels
                }

                test_p = [v["predictions"] for v in test_pred_str.values()]
                test_r = [v["predictions"] for v in test_ref_str.values()]

                # Match stet lists to test eids
                matched_stets = []
                for eid in test_pred_str:
                    if eid in test_data_text:
                        matched_stets.append(test_data_text[eid].get("stet", []))

                test_metrics = compute_all_metrics(test_p, test_r, matched_stets)
                print(f"[TEST] F1={test_metrics['F1']:.2f}, F1@30={test_metrics['F1@30']:.2f}, "
                      f"BS@30={test_metrics['BS@30']:.2f}, tIoU={test_metrics['tIoU']:.2f}")
                writer.add_scalars("eval/test", test_metrics, epoch)

                # Save best checkpoint
                if test_metrics["F1"] > best_f1:
                    best_f1 = test_metrics["F1"]
                    best_dir = os.path.join(OUTPUT_DIR, "best")
                    os.makedirs(best_dir, exist_ok=True)
                    model.get_llm_for_save().save_pretrained(os.path.join(best_dir, "lora_adapter"))
                    torch.save(
                        model.classifier.state_dict(),
                        os.path.join(best_dir, "classifier.pt"),
                    )
                    tokenizer.save_pretrained(best_dir)
                    print(f"[SAVE] best F1={best_f1:.2f} -> {best_dir}")

                # Save metrics
                metrics_summary = {
                    "epoch": epoch + 1,
                    "dev": {k: round(v, 2) for k, v in dev_metrics.items()},
                    "test": {k: round(v, 2) for k, v in test_metrics.items()},
                }
                with open(os.path.join(OUTPUT_DIR, f"metrics_epoch{epoch+1}.json"), "w") as f:
                    json.dump(metrics_summary, f, indent=2)

                # Save boundary predictions for generate pipeline
                pred_boundary_path = os.path.join(OUTPUT_DIR, "test_boundaries.jsonl")
                save_boundaries(test_pred_str, pred_boundary_path)
            except Exception as e:
                print(f"[EVAL ERROR] epoch={epoch+1}: {e}")
                import traceback
                traceback.print_exc()

    # ---- Final save ----
    final_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.get_llm_for_save().save_pretrained(os.path.join(final_dir, "lora_adapter"))
    torch.save(model.classifier.state_dict(), os.path.join(final_dir, "classifier.pt"))
    tokenizer.save_pretrained(final_dir)

    # Save final boundary predictions
    final_preds = get_predictions(model, test_dataset, tokenizer, device)
    final_pred_str = {
        eid: {"predictions": int_to_label(v["predictions"])}
        for eid, v in final_preds.items()
    }
    save_boundaries(final_pred_str, os.path.join(OUTPUT_DIR, "test_boundaries.jsonl"))

    writer.close()
    print(f"\nTraining done. Best test F1={best_f1:.2f}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
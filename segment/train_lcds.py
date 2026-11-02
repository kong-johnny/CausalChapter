"""Train the manuscript LCDS reconstruction with train/dev-only selection.

The historical model is unavailable; this entry point produces a new run.
It never loads the test split while training or tuning decoding parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, LongformerConfig, get_cosine_schedule_with_warmup

from .decoding import decode_probabilities
from .lcds import LCDSBoundaryHead
from .time_metrics import evaluate_internal_boundaries
from .utils import int_to_label, save_boundaries


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["example_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate video IDs: {path}")
    return rows


class VideoInputs:
    def __init__(self, rows, tokenizer, visual_cache, max_tokens):
        self.rows = rows
        self.prepared = []
        files = {}
        for path in Path(visual_cache).glob("*.npy"):
            video, _, index = path.stem.rpartition("-")
            if index.isdigit():
                files.setdefault(video, {})[int(index)] = path
        for row in rows:
            eid = row["example_id"]
            candidates = [eid, eid.replace("@@", "_"), eid.replace("@@", "--"), eid.split("@@")[-1]]
            visual_paths = next((files[key] for key in candidates if key in files), {})
            visual = []
            for index in range(len(row["text"])):
                path = visual_paths.get(index)
                if path is None:
                    raise FileNotFoundError(f"Missing utterance visual feature: {eid}, {index}")
                feature = np.load(path, allow_pickle=False)
                feature = feature.reshape(-1, feature.shape[-1]).mean(axis=0)
                if feature.shape != (512,) or not np.isfinite(feature).all():
                    raise ValueError(f"Invalid visual feature: {path}, {feature.shape}")
                visual.append(feature)
            tokens = []
            for text in row["text"]:
                token_ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_tokens - 2]
                tokens.append([tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id])
            windows = []
            start = 0
            while start < len(tokens):
                ids, positions, indices = [], [], []
                stop = start
                while stop < len(tokens) and len(ids) + len(tokens[stop]) <= max_tokens:
                    ids.extend(tokens[stop])
                    positions.append(len(ids) - 1)
                    indices.append(stop)
                    stop += 1
                windows.append((torch.tensor(ids), positions, indices))
                if stop >= len(tokens):
                    break
                start = max(start + 1, stop - 1)
            self.prepared.append((windows, torch.tensor(np.asarray(visual), dtype=torch.float32)))


class LongformerLCDS(nn.Module):
    def __init__(self, base_model, hidden_size, window_size, freeze_encoder=False, encoder_config=None, revision=None):
        super().__init__()
        if encoder_config is None:
            self.encoder = AutoModel.from_pretrained(base_model, revision=revision)
        else:
            self.encoder = AutoModel.from_config(LongformerConfig.from_dict(encoder_config))
        self.head = LCDSBoundaryHead(self.encoder.config.hidden_size, 512, hidden_size, window_size)
        self.freeze_encoder = freeze_encoder
        self.feature_cache = {}
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        else:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, prepared, device):
        windows, visual = prepared
        cache_key = id(prepared)
        if self.freeze_encoder and cache_key in self.feature_cache:
            return self.head(self.feature_cache[cache_key].to(device), visual.to(device))
        if self.freeze_encoder:
            self.encoder.eval()
        vectors = []
        indices = []
        with torch.set_grad_enabled(self.training and not self.freeze_encoder):
            for ids, positions, absolute in windows:
                ids = ids[None].to(device)
                mask = torch.ones_like(ids)
                global_mask = torch.zeros_like(ids)
                global_mask[:, 0] = 1
                hidden = self.encoder(input_ids=ids, attention_mask=mask, global_attention_mask=global_mask).last_hidden_state[0]
                vectors.append(hidden[positions])
                indices.extend(absolute)
        vectors = torch.cat(vectors)
        index = torch.tensor(indices, device=device)
        count = visual.shape[0]
        features = vectors.new_zeros(count, vectors.shape[-1]).index_add(0, index, vectors)
        denominator = torch.bincount(index, minlength=count).clamp_min(1).to(vectors.dtype)
        features = features / denominator[:, None]
        if self.freeze_encoder:
            self.feature_cache[cache_key] = features.detach().float().cpu()
        return self.head(features.float(), visual.to(device))


@torch.inference_mode()
def collect_scores(model, inputs, device):
    model.eval()
    result = {}
    for row, prepared in zip(inputs.rows, inputs.prepared):
        output = model(prepared, device)
        result[row["example_id"]] = output["logits"].float().softmax(-1)[:, 1].cpu().tolist()
    return result


def tune_dev(rows, scores):
    best = None
    for threshold in (0.2, 0.35, 0.5, 0.65, 0.8):
        for min_gap in (0, 2, 4, 6, 8):
            predictions = {eid: decode_probabilities(values, threshold, min_gap) for eid, values in scores.items()}
            metrics = evaluate_internal_boundaries(rows, predictions)
            key = (metrics["F1@30s"], metrics["internal_boundary_F1"], -metrics["boundary_count_MAE"])
            if best is None or key > best[0]:
                best = key, {"threshold": threshold, "min_gap": min_gap}, metrics
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--base-model", default="allenai/longformer-base-4096")
    parser.add_argument("--base-revision", default="301e6a42cb0d9976a6d6a26a079fef81c18aa895")
    parser.add_argument("--data-dir", default="data/avlecture")
    parser.add_argument("--visual-cache", default="data/avlecture/vis2d_cache/avlecture")
    parser.add_argument("--output", default="outputs/lcds_reconstruction")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--warm-start", help="A new LCDS checkpoint from a preceding training phase")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "evaluate":
        saved = json.loads((output / "best_config.json").read_text())
        for key in ("base_model", "hidden_size", "window_size", "max_tokens", "freeze_encoder"):
            setattr(args, key, saved[key])
    tokenizer_source = output / "tokenizer" if args.mode == "evaluate" else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=args.base_revision if args.mode == "train" else None)
    model = LongformerLCDS(
        args.base_model, args.hidden_size, args.window_size, args.freeze_encoder,
        encoder_config=saved["encoder_config"] if args.mode == "evaluate" else None,
        revision=args.base_revision,
    ).to(device)
    if args.mode == "train" and args.warm_start:
        model.load_state_dict(torch.load(args.warm_start, map_location="cpu", weights_only=True), strict=True)

    if args.mode == "evaluate":
        model.load_state_dict(torch.load(output / "best.pt", map_location="cpu", weights_only=True), strict=True)
        rows = read_rows(Path(args.data_dir) / f"{args.split}.jsonl")
        inputs = VideoInputs(rows, tokenizer, args.visual_cache, args.max_tokens)
        scores = collect_scores(model, inputs, device)
        predictions = {eid: decode_probabilities(values, saved["decoding"]["threshold"], saved["decoding"]["min_gap"]) for eid, values in scores.items()}
        metrics = evaluate_internal_boundaries(rows, predictions)
        (output / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        save_boundaries({eid: {"predictions": int_to_label(labels)} for eid, labels in predictions.items()}, str(output / f"{args.split}_boundaries.jsonl"))
        print(json.dumps(metrics, indent=2), flush=True)
        return

    train_path, dev_path = Path(args.data_dir) / "train.jsonl", Path(args.data_dir) / "dev.jsonl"
    train_rows, dev_rows = read_rows(train_path), read_rows(dev_path)
    if {row["example_id"] for row in train_rows} & {row["example_id"] for row in dev_rows}:
        raise ValueError("Training and development video IDs overlap")
    train = VideoInputs(train_rows, tokenizer, args.visual_cache, args.max_tokens)
    dev = VideoInputs(dev_rows, tokenizer, args.visual_cache, args.max_tokens)
    positives = sum(sum(row["labels"][:-1]) for row in train_rows)
    total = sum(len(row["labels"]) - 1 for row in train_rows)
    positive_fraction = max(positives / max(total, 1), 0.001)
    weights = torch.tensor([positive_fraction, 1 - positive_fraction], device=device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr)
    steps = math.ceil(len(train_rows) / args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.03 * steps), steps)
    provenance = {**vars(args), "status": "new manuscript reconstruction", "encoder_config": model.encoder.config.to_dict(), "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(), "dev_sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest()}
    tokenizer.save_pretrained(output / "tokenizer")
    (output / "run_config.json").write_text(json.dumps(provenance, indent=2) + "\n")
    best_key = (-1, -1, -float("inf"))
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        total_loss = 0.0
        for step, index in enumerate(order):
            row = train_rows[index]
            result = model(train.prepared[index], device)
            target = torch.tensor(row["labels"][:-1], device=device)
            loss = F.cross_entropy(result["logits"][:-1], target, weight=weights) + result["reconstruction_loss"]
            group_start = (step // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(order) - group_start)
            (loss / group_size).backward()
            total_loss += float(loss.detach())
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % 10 == 0:
                print(f"epoch={epoch + 1} video={step + 1}/{len(order)} loss={float(loss.detach()):.5f}", flush=True)
        scores = collect_scores(model, dev, device)
        key, decoding, metrics = tune_dev(dev_rows, scores)
        record = {"epoch": epoch + 1, "loss": total_loss / len(order), "dev": metrics, "decoding": decoding}
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if key > best_key:
            best_key = key
            torch.save(model.state_dict(), output / "best.pt.tmp")
            os.replace(output / "best.pt.tmp", output / "best.pt")
            config = {**provenance, "epoch": epoch + 1, "decoding": decoding, "dev_metrics": metrics}
            (output / "best_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("TRAINING_COMPLETE; test split has not been loaded", flush=True)


if __name__ == "__main__":
    main()

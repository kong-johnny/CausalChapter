"""Fresh-base, train-only two-pass CSSE training. No legacy train.py import.

    python -m generate.train_clean --help

Predicted train AND dev boundaries/timelines are mandatory. GT titles are
supervision/scoring only: never used to cut, prefilter, or prompt generation.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

from .data.clean_training import prediction_samples, supervised_predictions, validate_prediction_artifacts
from .data.split_audit import audited_train_rows, load_rows


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", required=True)
    result.add_argument("--base-model", required=True)
    result.add_argument("--visual-cache", required=True)
    result.add_argument("--train-boundaries", required=True)
    result.add_argument("--train-times", required=True)
    result.add_argument("--dev-boundaries", required=True)
    result.add_argument("--dev-times", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--fusion", choices=("text_only", "cross_attn"), default="text_only")
    result.add_argument("--epochs", type=int, default=3)
    result.add_argument("--lr", type=float, default=2e-5)
    result.add_argument("--grad-accum", type=int, default=4)
    result.add_argument("--max-length", type=int, default=4096)
    result.add_argument("--input-tokens", type=int, default=8192)
    result.add_argument("--rank", type=int, default=32)
    result.add_argument("--beams", type=int, default=3)
    result.add_argument("--topk", type=int, default=3)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--caption-matching", choices=("temporal", "lexical"), default="lexical")
    result.add_argument("--device", default="cuda")
    result.add_argument("--audit-only", action="store_true", help="Validate inputs without importing ML dependencies")
    result.add_argument("--smoke-steps", type=int, default=0, help="One train video, both passes, no dev selection or checkpoint")
    return result


def audit(args):
    data = Path(args.data_dir)
    sources = {"train_text": data / "train.jsonl", "train_gt": data / "train_subheading_abstract.jsonl",
               "dev_text": data / "dev.jsonl", "dev_gt": data / "dev_subheading_abstract.jsonl",
               "train_boundaries": Path(args.train_boundaries), "train_times": Path(args.train_times),
               "dev_boundaries": Path(args.dev_boundaries), "dev_times": Path(args.dev_times)}
    train, _, receipt = audited_train_rows(sources["train_text"], sources["train_gt"], sources["dev_text"])
    for split, rows in (("train", train), ("dev", load_rows(sources["dev_text"]))):
        expected = {row["example_id"] for row in rows}
        for role in ("gt", "boundaries", "times"):
            actual = []
            for row in load_rows(sources[f"{split}_{role}"]):
                eid = row["example_id"]
                if isinstance(eid, list):
                    if not eid or any(value != eid[0] for value in eid):
                        raise ValueError("Inconsistent repeated video ID")
                    eid = eid[0]
                actual.append(eid)
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError(f"{split} {role} ID set differs from its split")
        validate_prediction_artifacts(rows, load_rows(sources[f"{split}_boundaries"]),
                                      load_rows(sources[f"{split}_times"]))
    base = Path(args.base_model)
    if not (base / "config.json").is_file() or (base / "adapter_config.json").exists():
        raise ValueError("Require a complete local base model, not an inherited adapter")
    config = json.loads((base / "config.json").read_text())
    if config.get("model_type") != "qwen2":
        raise ValueError("This controlled experiment requires the Qwen2 backbone")
    weights = sorted(base.glob("*.safetensors"))
    if not weights:
        raise ValueError("No base safetensors found")
    for name in ("epochs", "grad_accum", "max_length", "input_tokens", "rank", "beams", "topk"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.lr <= 0 or args.smoke_steps < 0:
        raise ValueError("Invalid learning rate or smoke limit")
    receipt.update({"input_sha256": {str(path): digest(path) for path in sources.values()},
                    "base_sha256": {str(path): digest(path) for path in [base / "config.json", *weights]},
                    "base_provenance": "Local base bytes recorded; verify against official source before release.",
                    "initialization": "fresh base + new LoRA; no adapter warm-start",
                    "generation_segments": "all predicted train/dev chapters; references are loss/scoring only"})
    return sources, receipt


def main():
    args = parser().parse_args()
    sources, receipt = audit(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training_config.json").exists():
        raise ValueError("Refusing to overwrite/restart an existing experiment; this is not exact resume")
    (output / "input_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if args.audit_only:
        print("CLEAN_GENERATOR_INPUT_AUDIT_COMPLETE; no model loaded")
        return
    os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", CC_SEMANTIC_BACKEND="hashing")
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from .data.collator import JointCollator
    from .data.dataset import JointDataset
    from .data.eval_data_builder import build_strict_pred_eval_data
    from .data.tokenizer_compat import tokenizer_compat_kwargs
    from .data.vis_features import build_vis_index
    from .evaluation.evaluator import eval_model
    from .inference.pass2_builder import build_pass2_causal_samples
    from .models.model_builder import build_multimodal_model
    from .retrieval.kl_support_retriever import KLSupportRetriever

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/PPU unavailable; initialize SDK and intended environment first")
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_items = build_strict_pred_eval_data(sources["train_text"], sources["train_gt"],
                                             sources["train_boundaries"], sources["train_times"])
    dev_items = build_strict_pred_eval_data(sources["dev_text"], sources["dev_gt"],
                                           sources["dev_boundaries"], sources["dev_times"])
    for item in dev_items:
        item["caption_matching"] = args.caption_matching
    if args.smoke_steps:
        train_items = train_items[:1]
    raw_train = prediction_samples(train_items)
    train_samples = supervised_predictions(raw_train, train_items)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True,
                                              **tokenizer_compat_kwargs(args.base_model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    queries = 8 if args.fusion == "cross_attn" else 0
    if queries:
        tokenizer.add_tokens([f"<VIS_{i}>" for i in range(queries)], special_tokens=True)
    model, _ = build_multimodal_model(args.base_model, args.base_model, tokenizer,
                                      lora_r=args.rank, lora_alpha=args.rank * 2,
                                      gradient_checkpointing=True, fusion_mode=args.fusion,
                                      num_query_tokens=queries, attn_implementation="sdpa")
    model.to(device)
    model.llm.config.use_cache = False
    visual_index = build_vis_index(args.visual_cache) if queries else {}
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    retriever = KLSupportRetriever(model, tokenizer, device=str(device), near_size=3, semantic_size=10,
                                  final_topk=args.topk, intervention_tokens=16, variant_microbatch=2,
                                  vis_dim=512, vis_token_count=queries, input_max_tokens=args.input_tokens)
    generation = {"num_beams": args.beams, "repetition_penalty": 1.005, "no_repeat_ngram_size": 3}
    common = dict(model_pass1=model, tokenizer=tokenizer, vis_index=visual_index,
                  retrieval_system=retriever, vis_dim=512, vis_token_count=queries,
                  pass1_seg_tokens=6000, input_max_tokens=args.input_tokens,
                  use_fewshot=False, fewshot_num=0, device=str(device), gen_kwargs=generation)
    plan = {**vars(args), **receipt, "training_predicted_chapters": len(raw_train),
            "loss_labeled_chapters": len(train_samples), "selection": "dev CIDEr primary, SODA_c tiebreak",
            "minimum_training_target_iou": 0.3, "dev_segmentation": "predicted only; no prefilter",
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")}}
    (output / "training_config.json").write_text(json.dumps(plan, indent=2) + "\n")

    def train_pass(samples, mode):
        dataset = JointDataset(samples, tokenizer, visual_index, max_length=args.max_length,
                               vis_token_count=queries, prompt_mode=mode, use_fewshot=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=JointCollator(tokenizer))
        model.train()
        model.llm.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        total, count = 0.0, 0
        steps = min(len(loader), args.smoke_steps) if args.smoke_steps else len(loader)
        for step, batch in enumerate(loader):
            if step >= steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            window = min(args.grad_accum, steps - (step // args.grad_accum) * args.grad_accum)
            result = model(**batch)
            loss = result.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {mode} loss at step {step}")
            (loss / window).backward()
            total += float(loss.detach())
            count += 1
            del result, loss
            if (step + 1) % args.grad_accum == 0 or step + 1 == steps:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 10 == 0 or step + 1 == steps:
                print(f"TRAIN {mode} step={step + 1}/{steps} mean_loss={total/count:.4f}", flush=True)
        return total / max(count, 1)

    best = (-math.inf, -math.inf)
    for epoch in range(1, (1 if args.smoke_steps else args.epochs) + 1):
        first_loss = train_pass(train_samples, "pass1")
        model.eval()
        model.llm.config.use_cache = True
        with torch.no_grad():
            # No target fields, reference titles or reference times are passed
            # to CSSE. ALL predicted chapters generate before loss matching.
            pass2_raw = build_pass2_causal_samples(raw_train, causal_topk=args.topk, **common)
        if len(pass2_raw) != len(raw_train):
            raise RuntimeError("CSSE did not retain every predicted training chapter")
        second_samples = supervised_predictions(pass2_raw, train_items)
        second_loss = train_pass(second_samples, "pass2")
        if args.smoke_steps:
            (output / "smoke_receipt.json").write_text(json.dumps({"pass1_loss": first_loss,
                "pass2_loss": second_loss, "checkpoint_saved": False, "dev_scored": False,
                "all_training_predictions_generated": len(pass2_raw) == len(raw_train)}, indent=2) + "\n")
            print("CLEAN_GENERATOR_SMOKE_COMPLETE; not a full training result", flush=True)
            return
        model.eval()
        model.llm.config.use_cache = True
        with torch.no_grad():
            cider, meteor, soda = eval_model(model_pass2=model, eval_data=dev_items, topk=args.topk,
                use_fewshot=False, save_dir=str(output), boundary_type="dev_pred", epoch=epoch,
                **{key: value for key, value in common.items() if key != "use_fewshot"})
        metrics = {"epoch": epoch, "pass1_loss": first_loss, "pass2_loss": second_loss,
                   "CIDEr": cider, "METEOR": meteor, "SODA_c": soda,
                   "split": "dev", "protocol": "strict", "caption_matching": args.caption_matching}
        (output / f"metrics_epoch{epoch}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if (cider, soda) > best:
            stage, previous = output / "best_staging", output / "previous_best"
            if stage.exists() or previous.exists():
                raise RuntimeError("Prior incomplete checkpoint transaction requires inspection")
            stage.mkdir()
            model.llm.save_pretrained(stage / "shared_lora")
            tokenizer.save_pretrained(stage)
            if queries:
                torch.save(model.qformer.state_dict(), stage / "qformer.pt")
            (stage / "multimodal_meta.json").write_text(json.dumps({"base_model": args.base_model,
                "fusion_mode": args.fusion, "num_query_tokens": queries, "causal_mode": "light",
                "epoch": epoch, "selection_split": "dev", "best_pred_boundary": metrics}, indent=2) + "\n")
            (stage / "training_config.json").write_text(json.dumps(plan, indent=2) + "\n")
            best_path = output / "best"
            if best_path.exists():
                best_path.rename(previous)
            stage.rename(best_path)
            if previous.exists():
                shutil.rmtree(previous)  # Exact trainer-owned superseded best only.
            best = cider, soda
        print(json.dumps(metrics), flush=True)
    print("CLEAN_GENERATOR_TRAINING_COMPLETE; train gradients/dev selection; no test/oracle", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CausalChapter Training (Text-Only Mode)

Joint two-pass training for video chapter title generation using Qwen2.5-3B.
- Pass1: Generate initial chapter titles from segment text
- Pass2: Refine titles using causal context (KL-CSSE support selection)
- Fusion mode: text_only (no visual features)
- Causal mode: light (KL-based support selection)
"""

import os
import sys
import json
from pathlib import Path

# Allow running from project root: python generate/train.py
if __name__ == "__main__" and __package__ is None:
    __package__ = "generate"
    _THIS_FILE = Path(__file__).resolve()
    _PROJECT_ROOT = _THIS_FILE.parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from .config.paths import (
    TRAIN_GT_JSONL, DEV_GT_JSONL, TEST_GT_JSONL,
    TRAIN_MIX_JSONL, DEV_MIX_JSONL,
    TRAIN_TEXT_JSONL, DEV_TEXT_JSONL, TEST_TEXT_JSONL,
    TRAIN_MIX_TEXT_JSONL, DEV_MIX_TEXT_JSONL,
    TEST_LABEL_JSON, TEST_PRED_BOUNDARY, VIS_CACHE_DIR,
)
from .data.vis_features import build_vis_index
from .data.sample_builder import build_training_samples
from .data.eval_data_builder import build_gt_eval_data, build_pred_eval_data
from .data.dataset import JointDataset
from .data.collator import JointCollator
from .models.model_builder import build_multimodal_model
from .inference.pass2_builder import build_pass2_causal_samples
from .evaluation.evaluator import eval_model

# ═══════════════════════════════════════════════════════════════════
# Hardcoded configuration — reproduce exact scores
# ═══════════════════════════════════════════════════════════════════
BASE_MODEL = "models/Qwen2.5-3B-Instruct"  # local path
OUTPUT_DIR = "output"
TRAIN_MODE = "gt_only"

FUSION_MODE = "text_only"     # pure text, no visual features
CAUSAL_MODE = "light"         # KL-CSSE support selection (required)
CAUSAL_TOPK = 3

EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
LR = 5e-5
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.0
LABEL_SMOOTHING = 0.0

LORA_R = 64
LORA_ALPHA = 128

MAX_LENGTH = 4096
SEG_LEN = 8000
WINDOW_SIZE = 11
INPUT_MAX_TOKENS = 8192
PASS1_SEG_TOKENS = 6000
PASS2_SEG_TOKENS = 6000

VIS_DIM = 512
Q_HIDDEN = 512
NUM_QUERY_TOKENS = 0  # text_only: no VIS tokens
QFORMER_HEADS = 8
QFORMER_LAYERS = 2

MATCH_ALPHA_TIME = 0.35
MATCH_ALPHA_UTT = 0.35
MATCH_ALPHA_TEXT = 0.30
MIN_MATCH_SCORE = 0.0

USE_FEWSHOT = False
FEWSHOT_NUM = 3

GEN_NUM_BEAMS = 1
GEN_REPETITION_PENALTY = 1.15

LOG_INTERVAL = 10
EVAL_EVERY = 1
TENSORBOARD_DIR = None

SEED = 42

# Optional warm-start / freezing knobs for visual-fusion continuation runs.
INIT_LORA_ADAPTER = os.environ.get("CC_INIT_LORA_ADAPTER", "")
INIT_QFORMER = os.environ.get("CC_INIT_QFORMER", "")
TRAIN_QFORMER_ONLY = os.environ.get("CC_TRAIN_QFORMER_ONLY", "0").lower() in {"1", "true", "yes"}
TRAIN_LORA_ONLY = os.environ.get("CC_TRAIN_LORA_ONLY", "0").lower() in {"1", "true", "yes"}
# Optional param-group LRs for joint visual continuation runs.
# Defaults keep the old single-LR behavior; wrappers can set qformer high / LoRA low.
QFORMER_LR = float(os.environ.get("CC_QFORMER_LR", "0") or 0)
LORA_LR = float(os.environ.get("CC_LORA_LR", "0") or 0)
# Fast iteration knob: train/eval after pass1 without rebuilding/training pass2.
# In eval we still run the normal two-pass generator using the shared current model.
SKIP_PASS2_TRAIN = os.environ.get("CC_SKIP_PASS2_TRAIN", "0").lower() in {"1", "true", "yes"}
TRAIN_SAMPLE_LIMIT = int(os.environ.get("CC_TRAIN_SAMPLE_LIMIT", "0") or 0)
# ═══════════════════════════════════════════════════════════════════


def main():
    # ---- Seed for reproducibility ----
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # TensorBoard
    tb_dir = TENSORBOARD_DIR if TENSORBOARD_DIR else os.path.join(OUTPUT_DIR, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"[TB] TensorBoard logs -> {tb_dir}")

    print("=" * 80)
    print("CausalChapter Joint Training")
    print(f"base_model={BASE_MODEL}")
    print(f"fusion_mode={FUSION_MODE}")
    print(f"causal_mode={CAUSAL_MODE}")
    print(f"output_dir={OUTPUT_DIR}")
    print(f"lora_r={LORA_R}, lora_alpha={LORA_ALPHA}")
    print(f"lr={LR}, epochs={EPOCHS}")
    print("=" * 80)

    # 1) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # VIS token setup: text_only uses none; visual fusion modes reserve
    # explicit placeholders that are replaced by QFormer outputs.
    if FUSION_MODE == "text_only" or NUM_QUERY_TOKENS <= 0:
        vis_token_ids = []
        print("[INFO] text_only mode: skipping VIS token setup")
    else:
        vis_tokens = [f"<VIS_{i}>" for i in range(NUM_QUERY_TOKENS)]
        missing = [tok for tok in vis_tokens if tok not in tokenizer.get_vocab()]
        if missing:
            tokenizer.add_tokens(missing, special_tokens=True)
            print(f"[TOKENIZER] added VIS tokens: {missing}")
        vis_token_ids = tokenizer.convert_tokens_to_ids(vis_tokens)
        print(f"[INFO] visual fusion: vis_token_ids={vis_token_ids}")

    # 2) vis index (build empty one for text_only compatibility)
    print("\n[1/6] build visual index (empty for text_only)")
    vis_index = build_vis_index(VIS_CACHE_DIR)
    print(f"[VIS] indexed videos: {len(vis_index)}")

    # 3) training samples
    print("\n[2/6] build training samples")
    train_samples_gt = build_training_samples(
        train_mode=TRAIN_MODE,
        train_gt_jsonl=TRAIN_GT_JSONL,
        dev_gt_jsonl=DEV_GT_JSONL,
        train_text_jsonl=TRAIN_TEXT_JSONL,
        dev_text_jsonl=DEV_TEXT_JSONL,
        train_pred_boundary=None,
        dev_pred_boundary=None,
        alpha_time=MATCH_ALPHA_TIME,
        alpha_utt=MATCH_ALPHA_UTT,
        alpha_text=MATCH_ALPHA_TEXT,
        min_match_score=MIN_MATCH_SCORE,
    )
    for s in train_samples_gt:
        s["source"] = "gt"
    if TRAIN_SAMPLE_LIMIT > 0:
        print(f"[DATA] applying CC_TRAIN_SAMPLE_LIMIT={TRAIN_SAMPLE_LIMIT} (before={len(train_samples_gt)})")
        train_samples_gt = train_samples_gt[:TRAIN_SAMPLE_LIMIT]
    train_samples = train_samples_gt

    # 4) dataset
    print("\n[3/6] build pass1 dataset/loader")
    collator = JointCollator(tokenizer, vis_dim=VIS_DIM)

    train_dataset_pass1 = JointDataset(
        samples=train_samples, tokenizer=tokenizer, vis_index=vis_index,
        vis_dim=VIS_DIM, max_length=MAX_LENGTH, seg_len=SEG_LEN,
        window_size=WINDOW_SIZE, vis_token_count=NUM_QUERY_TOKENS,
        prompt_mode="pass1", use_fewshot=USE_FEWSHOT, fewshot_num=FEWSHOT_NUM,
    )
    train_loader_pass1 = DataLoader(
        train_dataset_pass1, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collator,
    )
    print(f"[DATASET] pass1 samples={len(train_dataset_pass1)}")

    # 5) model
    print("\n[4/6] load model")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_shared, _ = build_multimodal_model(
        base_model=BASE_MODEL,
        base_model_name=BASE_MODEL,
        tokenizer=tokenizer,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        gradient_checkpointing=False,
        use_4bit=False,
        use_8bit=False,
        fusion_mode=FUSION_MODE,
        num_query_tokens=NUM_QUERY_TOKENS,
        vis_hidden=VIS_DIM,
        attn_implementation="sdpa",
    )
    model_pass1 = model_shared.to(device)
    model_pass2 = model_shared

    # Optional: initialize LoRA from a strong text-only checkpoint, then learn
    # only the visual soft-prompt/QFormer (or continue joint LoRA+QFormer).
    if INIT_LORA_ADAPTER:
        from safetensors.torch import load_file
        from peft.utils import set_peft_model_state_dict
        adapter_file = INIT_LORA_ADAPTER
        if os.path.isdir(adapter_file):
            if os.path.exists(os.path.join(adapter_file, "shared_lora", "adapter_model.safetensors")):
                adapter_file = os.path.join(adapter_file, "shared_lora", "adapter_model.safetensors")
            else:
                adapter_file = os.path.join(adapter_file, "adapter_model.safetensors")
        print(f"[INIT] loading LoRA warm-start from {adapter_file}")
        state = load_file(adapter_file)
        incompatible = set_peft_model_state_dict(model_pass1.llm, state, adapter_name="default")
        miss = len(getattr(incompatible, "missing_keys", []) or [])
        unexp = len(getattr(incompatible, "unexpected_keys", []) or [])
        print(f"[INIT] LoRA warm-start done; missing={miss}, unexpected={unexp}")

    if INIT_QFORMER and getattr(model_pass1, "qformer", None) is not None:
        qformer_file = INIT_QFORMER
        if os.path.isdir(qformer_file):
            qformer_file = os.path.join(qformer_file, "qformer.pt")
        print(f"[INIT] loading Q-Former from {qformer_file}")
        model_pass1.qformer.load_state_dict(torch.load(qformer_file, map_location="cpu"), strict=True)

    if TRAIN_QFORMER_ONLY:
        print("[FREEZE] TRAIN_QFORMER_ONLY=1: freeze LLM/LoRA, train Q-Former only")
        for p in model_pass1.llm.parameters():
            p.requires_grad = False
        if getattr(model_pass1, "qformer", None) is not None:
            for p in model_pass1.qformer.parameters():
                p.requires_grad = True
    elif TRAIN_LORA_ONLY:
        print("[FREEZE] TRAIN_LORA_ONLY=1: freeze Q-Former, train LoRA only")
        if getattr(model_pass1, "qformer", None) is not None:
            for p in model_pass1.qformer.parameters():
                p.requires_grad = False

    # 6) optimizer
    print("\n[5/6] build optimizer")
    named_trainable_shared = [(n, p) for n, p in model_pass1.named_parameters() if p.requires_grad]
    trainable_shared = [p for _, p in named_trainable_shared]
    if QFORMER_LR > 0 or LORA_LR > 0:
        q_lr = QFORMER_LR if QFORMER_LR > 0 else LR
        l_lr = LORA_LR if LORA_LR > 0 else LR
        qformer_params = [(n, p) for n, p in named_trainable_shared if n.startswith("qformer.") or ".qformer." in n]
        other_params = [(n, p) for n, p in named_trainable_shared if not (n.startswith("qformer.") or ".qformer." in n)]
        param_groups = []
        if qformer_params:
            param_groups.append({"params": [p for _, p in qformer_params], "lr": q_lr, "weight_decay": WEIGHT_DECAY})
        if other_params:
            param_groups.append({"params": [p for _, p in other_params], "lr": l_lr, "weight_decay": WEIGHT_DECAY})
        optimizer_shared = torch.optim.AdamW(param_groups)
        print(f"[OPT] param groups: qformer={sum(p.numel() for _, p in qformer_params):,} lr={q_lr:g}; "
              f"other/lora={sum(p.numel() for _, p in other_params):,} lr={l_lr:g}")
    else:
        optimizer_shared = torch.optim.AdamW(trainable_shared, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_shared = None
    print(f"[PARAM] shared trainable={sum(p.numel() for p in trainable_shared):,}")
    print("[SCHED] disabled (warmup_ratio=0)")

    # 7) prepare eval data
    print("\n[6/6] prepare eval data")
    gt_eval_data = build_gt_eval_data(TEST_GT_JSONL, TEST_TEXT_JSONL)
    pred_eval_data = build_pred_eval_data(TEST_TEXT_JSONL, TEST_GT_JSONL, TEST_PRED_BOUNDARY)

    # 8) retrieval system (causal mode)
    retrieval_system = None
    if CAUSAL_MODE != "off":
        from .retrieval.kl_support_retriever import KLSupportRetriever
        retrieval_system = KLSupportRetriever(
            generation_model=model_pass2,
            tokenizer=tokenizer,
            device=device,
            near_size=3,
            semantic_size=10,
            final_topk=CAUSAL_TOPK,
            intervention_tokens=16,
            variant_microbatch=2,
            vis_dim=VIS_DIM,
            vis_token_count=NUM_QUERY_TOKENS,
            input_max_tokens=INPUT_MAX_TOKENS,
        )
        print("[CAUSAL] enabled mode=light (KL-CSSE)")
    else:
        print("[CAUSAL] disabled")

    # Generation kwargs
    gen_kwargs = {"repetition_penalty": GEN_REPETITION_PENALTY}
    if GEN_NUM_BEAMS > 1:
        gen_kwargs["num_beams"] = GEN_NUM_BEAMS
    print(f"[GEN] kwargs: {gen_kwargs}")

    # ============ Training ============
    print("\n" + "=" * 60)
    print("START TRAINING")
    print("=" * 60)
    best_cider = -1.0
    best_meteor = -1.0
    best_soda = -1.0

    for epoch in range(EPOCHS):
        # ---- Pass1 training ----
        model_pass1.train()
        total_loss_pass1 = 0.0
        step_cnt_pass1 = 0
        accum_cnt = 0

        pbar = tqdm(train_loader_pass1, desc=f"Epoch {epoch+1}/{EPOCHS} [pass1]")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            vis_seq = batch["vis_seq"].to(device)
            vis_mask = batch["vis_mask"].to(device)

            outputs = model_pass1(
                input_ids=input_ids, attention_mask=attention_mask,
                labels=labels, vis_seq=vis_seq, vis_mask=vis_mask,
            )
            loss = outputs.loss

            if torch.isnan(loss):
                optimizer_shared.zero_grad()
                continue

            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            accum_cnt += 1

            if accum_cnt == GRADIENT_ACCUMULATION_STEPS:
                torch.nn.utils.clip_grad_norm_(trainable_shared, 1.0)
                optimizer_shared.step()
                optimizer_shared.zero_grad()
                if scheduler_shared is not None:
                    scheduler_shared.step()
                accum_cnt = 0

            total_loss_pass1 += float(loss.item() * GRADIENT_ACCUMULATION_STEPS)
            step_cnt_pass1 += 1
            pbar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})

        # Flush remaining gradients
        if accum_cnt > 0:
            torch.nn.utils.clip_grad_norm_(trainable_shared, 1.0)
            optimizer_shared.step()
            optimizer_shared.zero_grad()
            accum_cnt = 0

        avg_loss_p1 = total_loss_pass1 / max(1, step_cnt_pass1)
        print(f"\n[TRAIN][pass1] epoch={epoch+1}, avg_loss={avg_loss_p1:.4f}")
        writer.add_scalar("train/epoch_pass1_loss", avg_loss_p1, epoch)

        # ---- Build/train causal Pass2 samples ----
        if SKIP_PASS2_TRAIN:
            print("\n[BUILD] skip pass2 causal training samples (CC_SKIP_PASS2_TRAIN=1)")
            avg_loss_p2 = float("nan")
            writer.add_scalar("train/epoch_pass2_loss", avg_loss_p2, epoch)
        else:
            # ---- Build causal Pass2 samples ----
            print("\n[BUILD] refresh pass2 causal training samples")
            model_pass1.eval()
            train_samples_pass2 = build_pass2_causal_samples(
                samples=[{**s, "source": "gt"} for s in train_samples_gt],
                model_pass1=model_pass1, tokenizer=tokenizer, vis_index=vis_index,
                retrieval_system=retrieval_system,
                vis_dim=VIS_DIM, vis_token_count=NUM_QUERY_TOKENS,
                causal_topk=CAUSAL_TOPK,
                pass1_seg_tokens=PASS1_SEG_TOKENS,
                input_max_tokens=INPUT_MAX_TOKENS,
                use_fewshot=USE_FEWSHOT, fewshot_num=FEWSHOT_NUM,
                device=device, gen_kwargs=gen_kwargs,
            )

            train_dataset_pass2 = JointDataset(
                samples=train_samples_pass2, tokenizer=tokenizer, vis_index=vis_index,
                vis_dim=VIS_DIM, max_length=MAX_LENGTH, seg_len=SEG_LEN,
                window_size=WINDOW_SIZE, vis_token_count=NUM_QUERY_TOKENS,
                prompt_mode="pass2", use_fewshot=USE_FEWSHOT, fewshot_num=FEWSHOT_NUM,
            )
            train_loader_pass2 = DataLoader(
                train_dataset_pass2, batch_size=BATCH_SIZE,
                shuffle=True, collate_fn=collator,
            )
            print(f"[DATASET] pass2 causal samples={len(train_dataset_pass2)}")

            # ---- Pass2 training ----
            model_pass2.train()
            total_loss_pass2 = 0.0
            step_cnt_pass2 = 0
            accum_cnt = 0

            pbar = tqdm(train_loader_pass2, desc=f"Epoch {epoch+1}/{EPOCHS} [pass2]")
            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                vis_seq = batch["vis_seq"].to(device)
                vis_mask = batch["vis_mask"].to(device)

                outputs = model_pass2(
                    input_ids=input_ids, attention_mask=attention_mask,
                    labels=labels, vis_seq=vis_seq, vis_mask=vis_mask,
                )
                loss = outputs.loss

                if torch.isnan(loss):
                    optimizer_shared.zero_grad()
                    continue

                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                accum_cnt += 1

                if accum_cnt == GRADIENT_ACCUMULATION_STEPS:
                    torch.nn.utils.clip_grad_norm_(trainable_shared, 1.0)
                    optimizer_shared.step()
                    optimizer_shared.zero_grad()
                    if scheduler_shared is not None:
                        scheduler_shared.step()
                    accum_cnt = 0

                total_loss_pass2 += float(loss.item() * GRADIENT_ACCUMULATION_STEPS)
                step_cnt_pass2 += 1
                pbar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})

            # Flush remaining gradients
            if accum_cnt > 0:
                torch.nn.utils.clip_grad_norm_(trainable_shared, 1.0)
                optimizer_shared.step()
                optimizer_shared.zero_grad()
                accum_cnt = 0

            avg_loss_p2 = total_loss_pass2 / max(1, step_cnt_pass2)
            print(f"\n[TRAIN][pass2] epoch={epoch+1}, avg_loss={avg_loss_p2:.4f}")
            writer.add_scalar("train/epoch_pass2_loss", avg_loss_p2, epoch)

        # ---- Evaluation ----
        if (epoch + 1) % EVAL_EVERY == 0:
            model_pass1.eval()
            model_pass2.eval()

            print("\n===== TEST: GT boundaries (oracle) =====")
            cider_gt, meteor_gt, soda_gt = eval_model(
                model_pass1=model_pass1, model_pass2=model_pass2,
                tokenizer=tokenizer, eval_data=gt_eval_data, vis_index=vis_index,
                retrieval_system=retrieval_system,
                vis_dim=VIS_DIM, vis_token_count=NUM_QUERY_TOKENS,
                topk=CAUSAL_TOPK,
                pass1_seg_tokens=PASS1_SEG_TOKENS,
                pass2_seg_tokens=PASS2_SEG_TOKENS,
                input_max_tokens=INPUT_MAX_TOKENS,
                use_fewshot=USE_FEWSHOT, fewshot_num=FEWSHOT_NUM,
                device=device, gen_kwargs=gen_kwargs,
                save_dir=OUTPUT_DIR, boundary_type="gt", epoch=epoch,
            )
            print(f"[GT] CIDEr={cider_gt:.2f}, METEOR={meteor_gt:.2f}, SODA={soda_gt:.2f}")
            writer.add_scalars("eval/gt", {
                "CIDEr": cider_gt, "METEOR": meteor_gt, "SODA": soda_gt,
            }, epoch)

            print("\n===== TEST: Pred boundaries (E2E) =====")
            cider_pred, meteor_pred, soda_pred = eval_model(
                model_pass1=model_pass1, model_pass2=model_pass2,
                tokenizer=tokenizer, eval_data=pred_eval_data, vis_index=vis_index,
                retrieval_system=retrieval_system,
                vis_dim=VIS_DIM, vis_token_count=NUM_QUERY_TOKENS,
                topk=CAUSAL_TOPK,
                pass1_seg_tokens=PASS1_SEG_TOKENS,
                pass2_seg_tokens=PASS2_SEG_TOKENS,
                input_max_tokens=INPUT_MAX_TOKENS,
                use_fewshot=USE_FEWSHOT, fewshot_num=FEWSHOT_NUM,
                device=device, gen_kwargs=gen_kwargs,
                save_dir=OUTPUT_DIR, boundary_type="pred", epoch=epoch,
            )
            print(f"[Pred] CIDEr={cider_pred:.2f}, METEOR={meteor_pred:.2f}, SODA={soda_pred:.2f}")
            writer.add_scalars("eval/pred", {
                "CIDEr": cider_pred, "METEOR": meteor_pred, "SODA": soda_pred,
            }, epoch)

            metrics = {
                "epoch": epoch + 1,
                "causal_mode": CAUSAL_MODE,
                "pred_boundary": {"CIDEr": round(cider_pred, 2), "METEOR": round(meteor_pred, 2), "SODA": round(soda_pred, 2)},
                "gt_boundary": {"CIDEr": round(cider_gt, 2), "METEOR": round(meteor_gt, 2), "SODA": round(soda_gt, 2)},
            }
            with open(os.path.join(OUTPUT_DIR, f"metrics_epoch{epoch+1}.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            # Save best checkpoint (based on pred-boundary CIDEr + SODA)
            if (cider_pred > best_cider) and (soda_pred > best_soda):
                best_cider = max(best_cider, cider_pred)
                best_meteor = max(best_meteor, meteor_pred)
                best_soda = max(best_soda, soda_pred)

                best_dir = os.path.join(OUTPUT_DIR, "best")
                os.makedirs(best_dir, exist_ok=True)
                model_pass1.llm.save_pretrained(os.path.join(best_dir, "shared_lora"))
                if getattr(model_pass1, "qformer", None) is not None:
                    torch.save(model_pass1.qformer.state_dict(), os.path.join(best_dir, "qformer.pt"))
                tokenizer.save_pretrained(best_dir)

                meta = {
                    "base_model": BASE_MODEL,
                    "fusion_mode": FUSION_MODE,
                    "causal_mode": CAUSAL_MODE,
                    "num_query_tokens": NUM_QUERY_TOKENS,
                    "best_pred_boundary": {
                        "CIDEr": round(cider_pred, 2),
                        "METEOR": round(meteor_pred, 2),
                        "SODA": round(soda_pred, 2),
                    },
                }
                with open(os.path.join(best_dir, "multimodal_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"[SAVE] new best -> {best_dir}")

    # ---- Final save ----
    final_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    model_pass1.llm.save_pretrained(os.path.join(final_dir, "shared_lora"))
    if getattr(model_pass1, "qformer", None) is not None:
        torch.save(model_pass1.qformer.state_dict(), os.path.join(final_dir, "qformer.pt"))
    tokenizer.save_pretrained(final_dir)

    meta = {
        "base_model": BASE_MODEL,
        "fusion_mode": FUSION_MODE,
        "causal_mode": CAUSAL_MODE,
        "num_query_tokens": NUM_QUERY_TOKENS,
        "best_pred_boundary": {
            "CIDEr": round(best_cider, 2),
            "METEOR": round(best_meteor, 2),
            "SODA": round(best_soda, 2),
        },
    }
    with open(os.path.join(final_dir, "multimodal_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Fix adapter_config.json base_model path
    for saved_path in [os.path.join(OUTPUT_DIR, "best"), final_dir]:
        if not os.path.exists(saved_path):
            continue
        config_path = os.path.join(saved_path, "shared_lora", "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            if config.get("base_model_name_or_path") is None or config.get("base_model_name_or_path") != BASE_MODEL:
                config["base_model_name_or_path"] = BASE_MODEL
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                print(f"[FIXED] {config_path}")

    writer.close()
    print("\n" + "=" * 60)
    print("Training done.")
    print(f"Best Pred-boundary TEST: CIDEr={best_cider:.2f}, METEOR={best_meteor:.2f}, SODA={best_soda:.2f}")
    print(f"[TB] TensorBoard logs -> {tb_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
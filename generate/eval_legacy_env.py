# AUTO-GENERATED legacy env eval: original loader, env-configurable
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CausalChapter Evaluation

Load a trained checkpoint and evaluate CIDEr, METEOR, SODA_c on both
ground-truth and predicted boundaries.
"""
import os
import sys
import json
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    __package__ = "generate"
    _THIS_FILE = Path(__file__).resolve()
    _PROJECT_ROOT = _THIS_FILE.parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from transformers import AutoTokenizer
from peft import PeftModel

from .config.paths import (
    TEST_GT_JSONL, TEST_TEXT_JSONL, TEST_PRED_BOUNDARY, VIS_CACHE_DIR,
)
from .data.vis_features import build_vis_index
from .data.eval_data_builder import build_gt_eval_data, build_pred_eval_data
from .models.model_builder import build_multimodal_model
from .evaluation.evaluator import eval_model

# ═══════════════════════════════════════════════════════════════════
# Hardcoded configuration
# ═══════════════════════════════════════════════════════════════════
BASE_MODEL = os.environ.get("CC_BASE_MODEL", "models/Qwen2.5-3B-Instruct")
CHECKPOINT_DIR = os.environ.get("CC_CHECKPOINT_DIR", "output/best")
FUSION_MODE = os.environ.get("CC_FUSION_MODE", "text_only")
CAUSAL_MODE = os.environ.get("CC_CAUSAL_MODE", "light")
CAUSAL_TOPK = int(os.environ.get("CC_CAUSAL_TOPK", "3"))

LORA_R = int(os.environ.get("CC_LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("CC_LORA_ALPHA", "128"))

INPUT_MAX_TOKENS = int(os.environ.get("CC_INPUT_MAX_TOKENS", "8192"))
PASS1_SEG_TOKENS = int(os.environ.get("CC_PASS1_SEG_TOKENS", "6000"))
PASS2_SEG_TOKENS = int(os.environ.get("CC_PASS2_SEG_TOKENS", "6000"))
VIS_DIM = 512
NUM_QUERY_TOKENS = 0

USE_FEWSHOT = os.environ.get("CC_USE_FEWSHOT", "0").lower() in {"1", "true", "yes"}
FEWSHOT_NUM = int(os.environ.get("CC_FEWSHOT_NUM", "3"))

GEN_NUM_BEAMS = int(os.environ.get("CC_GEN_NUM_BEAMS", "1"))
GEN_REPETITION_PENALTY = float(os.environ.get("CC_GEN_REPETITION_PENALTY", "1.15"))
EVAL_SCOPE = os.environ.get("CC_EVAL_SCOPE", "both").lower()
EVAL_SAVE_DIR = os.environ.get("CC_EVAL_SAVE_DIR", CHECKPOINT_DIR)
TEST_PRED_BOUNDARY = os.environ.get("CC_TEST_PRED_BOUNDARY", TEST_PRED_BOUNDARY)
# ═══════════════════════════════════════════════════════════════════


def load_checkpoint(checkpoint_dir, base_model, tokenizer):
    """Load a trained LoRA adapter + tokenizer."""
    lora_path = os.path.join(checkpoint_dir, "shared_lora")
    if not os.path.exists(lora_path):
        lora_path = os.path.join(checkpoint_dir, "pass1_lora")

    if not os.path.exists(lora_path):
        raise FileNotFoundError(
            f"No LoRA adapter found in {checkpoint_dir}. "
            f"Expected 'shared_lora/' or 'pass1_lora/' subdirectory."
        )

    # Build a fresh multimodal model
    model, _ = build_multimodal_model(
        base_model=base_model,
        base_model_name=base_model,
        tokenizer=tokenizer,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        fusion_mode=FUSION_MODE,
        num_query_tokens=NUM_QUERY_TOKENS,
        vis_hidden=VIS_DIM,
        attn_implementation="sdpa",
    )

    # Load LoRA weights (resize embeds to match training vocab if needed)
    embed_shape = model.llm.get_input_embeddings().weight.shape
    meta_path = os.path.join(checkpoint_dir, "multimodal_meta.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
    model.llm = PeftModel.from_pretrained(model.llm, lora_path)
    print(f"[LOAD] LoRA adapter loaded from {lora_path}")

    return model, model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("CausalChapter Evaluation")
    print(f"checkpoint={CHECKPOINT_DIR}")
    print(f"fusion_mode={FUSION_MODE}")
    print("=" * 60)

    # Tokenizer: load from BASE_MODEL for vocab size consistency with training
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, local_files_only=True
    )
    print(f"[TOKENIZER] loaded from {BASE_MODEL} (vocab={len(tokenizer)})")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model_pass1, model_pass2 = load_checkpoint(CHECKPOINT_DIR, BASE_MODEL, tokenizer)
    model_pass1 = model_pass1.to(device)
    model_pass2 = model_pass2.to(device)
    model_pass1.eval()
    model_pass2.eval()

    # Data
    vis_index = build_vis_index(VIS_CACHE_DIR)
    gt_eval_data = build_gt_eval_data(TEST_GT_JSONL, TEST_TEXT_JSONL)
    pred_eval_data = build_pred_eval_data(TEST_TEXT_JSONL, TEST_GT_JSONL, TEST_PRED_BOUNDARY)

    # Retrieval
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
        print("[CAUSAL] enabled mode=light")

    gen_kwargs = {"repetition_penalty": GEN_REPETITION_PENALTY}
    if GEN_NUM_BEAMS > 1:
        gen_kwargs["num_beams"] = GEN_NUM_BEAMS

    cider_gt = meteor_gt = soda_gt = None
    cider_pred = meteor_pred = soda_pred = None
    os.makedirs(EVAL_SAVE_DIR, exist_ok=True)

    if EVAL_SCOPE in {"both", "gt"}:
        # ---- GT boundary evaluation ----
        print("\n" + "=" * 60)
        print("EVAL: Ground-Truth Boundaries (Oracle)")
        print("=" * 60)
        with torch.no_grad():
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
                save_dir=EVAL_SAVE_DIR, boundary_type="gt",
            )
        print(f"[GT] CIDEr={cider_gt:.2f}, METEOR={meteor_gt:.2f}, SODA_c={soda_gt:.2f}")

    if EVAL_SCOPE in {"both", "pred"}:
        # ---- Pred boundary evaluation (E2E) ----
        print("\n" + "=" * 60)
        print("EVAL: Predicted Boundaries (End-to-End)")
        print("=" * 60)
        with torch.no_grad():
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
                save_dir=EVAL_SAVE_DIR, boundary_type="pred",
            )
        print(f"[Pred] CIDEr={cider_pred:.2f}, METEOR={meteor_pred:.2f}, SODA_c={soda_pred:.2f}")

    if EVAL_SCOPE not in {"both", "gt", "pred"}:
        raise ValueError(f"Invalid EVAL_SCOPE={EVAL_SCOPE!r}; expected both|gt|pred")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if cider_gt is not None:
        print(f"  GT boundaries:    CIDEr={cider_gt:.2f}, METEOR={meteor_gt:.2f}, SODA_c={soda_gt:.2f}")
    else:
        print("  GT boundaries:    SKIPPED")
    if cider_pred is not None:
        print(f"  Pred boundaries:  CIDEr={cider_pred:.2f}, METEOR={meteor_pred:.2f}, SODA_c={soda_pred:.2f}")
    else:
        print("  Pred boundaries:  SKIPPED")

    # Target scores for reference
    print(f"\n  Target (pred):    CIDEr=104.59, METEOR=17.87, SODA_c=12.73")

    # Save summary
    summary = {
        "checkpoint": CHECKPOINT_DIR,
        "eval_scope": EVAL_SCOPE,
        "gt_boundary": None if cider_gt is None else {"CIDEr": round(cider_gt, 2), "METEOR": round(meteor_gt, 2), "SODA_c": round(soda_gt, 2)},
        "pred_boundary": None if cider_pred is None else {"CIDEr": round(cider_pred, 2), "METEOR": round(meteor_pred, 2), "SODA_c": round(soda_pred, 2)},
    }
    with open(os.path.join(EVAL_SAVE_DIR, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] eval_summary.json -> {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
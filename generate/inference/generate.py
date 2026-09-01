# -*- coding: utf-8 -*-
"""Single- and batched-generation helpers."""
from typing import Iterable, List, Optional

import numpy as np
import torch

from ..utils.text_utils import postprocess_title


def _as_vis_array(value, vis_dim):
    """Normalize visual sequence for generation.

    load_vis_seq() returns a single all-zero feature when CLIP is missing.
    Keep that as a *valid* visual sequence for visual-fusion models so QFormer
    replaces <VIS_i> consistently with training.  Otherwise missing-CLIP cases
    leave randomly initialized <VIS_i> token embeddings in the prompt.
    """
    if value is None:
        return np.zeros((1, vis_dim), dtype=np.float32), False
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[-1] != vis_dim:
        return np.zeros((1, vis_dim), dtype=np.float32), False
    return arr, True


def _pad_visual_batch(vis_sequences, vis_dim, device, disable_visual):
    if disable_visual:
        return None, None
    normalized = [_as_vis_array(value, vis_dim) for value in vis_sequences]
    # For cross-attn checkpoints the prompt contains <VIS_i> placeholders.
    # If an entire microbatch has missing CLIP, returning (None, None) leaves
    # those newly-added token embeddings in the text stream, which is a large
    # train/infer mismatch.  Instead, feed a one-step zero visual token with
    # mask=1 for invalid/missing clips, so QFormer always replaces <VIS_i>.
    max_t = max(arr.shape[0] for arr, _ in normalized)
    batch = np.zeros((len(normalized), max_t, vis_dim), dtype=np.float32)
    mask = np.zeros((len(normalized), max_t), dtype=np.int64)
    for row, (arr, valid) in enumerate(normalized):
        batch[row, : arr.shape[0]] = arr
        mask[row, : arr.shape[0]] = 1
    return (
        torch.from_numpy(batch).to(device),
        torch.from_numpy(mask).to(device),
    )


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    vis_sequences: Optional[Iterable],
    device,
    *,
    vis_dim=512,
    max_new_tokens=50,
    microbatch=2,
    disable_visual=None,
    return_token_ids=False,
    **gen_kwargs,
):
    """Generate a list of titles without applying a second hidden truncation."""
    if not prompts:
        return []
    if disable_visual is None:
        disable_visual = getattr(model, "disable_visual", False)
    vis_sequences = list(vis_sequences or [None] * len(prompts))
    if len(vis_sequences) != len(prompts):
        raise ValueError("prompts and vis_sequences must have equal length")

    generation = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": 1,
        "do_sample": False,
        "no_repeat_ngram_size": 3,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": False,
        "output_scores": False,
    }
    generation.update(gen_kwargs)

    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    results = [None] * len(prompts)
    # Length bucketing reduces left-padding waste while preserving the caller's
    # chapter order through indexed writes below.
    lengths = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in prompts
    ]
    order = sorted(range(len(prompts)), key=lambda idx: lengths[idx])
    try:
        for start in range(0, len(order), max(1, microbatch)):
            chunk_indices = order[start : start + microbatch]
            chunk_prompts = [prompts[idx] for idx in chunk_indices]
            chunk_vis = [vis_sequences[idx] for idx in chunk_indices]
            encoded = tokenizer(
                chunk_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            vis_seq, vis_mask = _pad_visual_batch(
                chunk_vis, vis_dim, device, disable_visual
            )
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    vis_seq=vis_seq,
                    vis_mask=vis_mask,
                    **generation,
                )

            # MultiModalCausalLM calls HF generate with inputs_embeds, so the
            # returned IDs are generated IDs rather than prompt+generated IDs.
            for original_idx, row in zip(chunk_indices, outputs):
                ids = row.detach().cpu()
                if return_token_ids:
                    results[original_idx] = ids
                else:
                    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                    results[original_idx] = postprocess_title(text)
    finally:
        tokenizer.padding_side = old_padding
    return results


def generate_one(
    model,
    tokenizer,
    prompt,
    vis_seq_np,
    device,
    debug=False,
    disable_visual=None,
    vis_dim=512,
    max_new_tokens=50,
    **gen_kwargs,
):
    result = generate_batch(
        model,
        tokenizer,
        [prompt],
        [vis_seq_np],
        device,
        vis_dim=vis_dim,
        max_new_tokens=max_new_tokens,
        microbatch=1,
        disable_visual=disable_visual,
        **gen_kwargs,
    )[0]
    if debug:
        print("[DEBUG] generated =", repr(result))
    return result

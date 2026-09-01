# -*- coding: utf-8 -*-
"""End-to-end paper-aligned CausalChapter inference."""
from __future__ import annotations

import math
import time

import torch

from ..config.fewshot import FEWSHOT_EXAMPLES
from ..data.prompt_builder import build_prompt_parts
from ..data.vis_features import load_vis_seq
from ..retrieval.kl_support_retriever import KLSupportRetriever
from ..utils.text_utils import build_prompt_with_truncation, postprocess_title
from .generate import generate_batch
from .pass1_inference import generate_pass1_titles_for_video


def _sync(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def causal_selection_inference(
    model_pass1,
    model_pass2,
    tokenizer,
    example_id,
    segments,
    spans,
    vis_index,
    times=None,
    retrieval_system=None,
    vis_dim=512,
    vis_token_count=4,
    candidate_size=10,
    topk=5,
    pass1_seg_tokens=6000,
    pass2_seg_tokens=4500,
    input_max_tokens=8192,
    use_fewshot=True,
    fewshot_num=3,
    device="cuda",
    gen_kwargs=None,
    pass_batch_size=2,
    variant_microbatch=2,
):
    """Run Pass 1, KL-CSSE, and Pass 2 for one video."""
    del pass2_seg_tokens
    if len(segments) != len(spans):
        raise ValueError("segments and spans must have equal length")
    video_samples = []
    for idx, (segment, span) in enumerate(zip(segments, spans)):
        sample = dict(segment)
        sample["start_idx"] = int(span[0])
        sample["end_idx"] = int(span[1])
        if times is not None and idx < len(times):
            sample["start_time"] = float(times[idx][0])
            sample["end_time"] = float(times[idx][1])
        video_samples.append(sample)

    visuals = [
        load_vis_seq(
            vis_index,
            example_id,
            sample["start_idx"],
            sample["end_idx"],
            vis_dim=vis_dim,
            start_time=sample.get("start_time"),
            end_time=sample.get("end_time"),
        )
        for sample in video_samples
    ]
    fewshot = FEWSHOT_EXAMPLES[:fewshot_num] if use_fewshot else None
    if not isinstance(retrieval_system, KLSupportRetriever):
        if retrieval_system is not None:
            print(
                "[CSSE] ignoring legacy retrieval_system; using KL support "
                "selection for the paper configuration"
            )
        retrieval_system = KLSupportRetriever(
            generation_model=model_pass2,
            tokenizer=tokenizer,
            device=device,
            near_size=3,
            semantic_size=candidate_size,
            final_topk=topk,
            intervention_tokens=16,
            variant_microbatch=variant_microbatch,
            vis_dim=vis_dim,
            vis_token_count=vis_token_count,
            input_max_tokens=input_max_tokens,
        )

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    retrieval_system.reset_stats()
    _sync(device)
    total_start = time.perf_counter()

    _sync(device)
    pass1_start = time.perf_counter()
    pass1_titles, pass1_info = generate_pass1_titles_for_video(
        model_pass1=model_pass1,
        tokenizer=tokenizer,
        example_id=example_id,
        video_samples=video_samples,
        vis_index=vis_index,
        vis_dim=vis_dim,
        vis_token_count=vis_token_count,
        pass1_seg_tokens=pass1_seg_tokens,
        input_max_tokens=input_max_tokens,
        use_fewshot=use_fewshot,
        fewshot_num=fewshot_num,
        device=device,
        gen_kwargs=gen_kwargs,
        batch_size=pass_batch_size,
        return_info=True,
    )
    _sync(device)
    pass1_time = time.perf_counter() - pass1_start

    retrieval_segments = []
    for idx, (sample, title) in enumerate(zip(video_samples, pass1_titles)):
        retrieval_segments.append(
            {
                "text": sample["segment_text"],
                "title": postprocess_title(title),
                "index": idx,
                "time_span": [sample["start_idx"], sample["end_idx"]],
            }
        )
    retrieval_system.ensure_built(retrieval_segments, video_key=example_id)

    selected_by_chapter = []
    support_details = []
    _sync(device)
    csse_start = time.perf_counter()
    for current_idx, current in enumerate(retrieval_segments):
        selected, details = retrieval_system.select_contexts(
            current_segment=current,
            all_segments=retrieval_segments,
            current_idx=current_idx,
            vis_seq_np=visuals[current_idx],
            candidate_size=candidate_size,
            topk=topk,
            fewshot_examples=fewshot,
            gen_kwargs=gen_kwargs,
        )
        selected_by_chapter.append(selected)
        support_details.append(details)
    _sync(device)
    csse_time = time.perf_counter() - csse_start

    pass2_prompts = []
    for idx, sample in enumerate(video_samples):
        context_titles = [
            pass1_titles[j] for j in selected_by_chapter[idx]
        ]
        head, middle, tail = build_prompt_parts(
            segment_text=sample["segment_text"],
            context_subheadings=context_titles,
            fewshot_examples=fewshot,
            max_context=topk,
            num_query_tokens=vis_token_count,
            pass1_self_title=pass1_titles[idx],
            tokenizer=tokenizer,
        )
        pass2_prompts.append(
            build_prompt_with_truncation(
                tokenizer,
                head,
                middle,
                tail,
                max_total_tokens=input_max_tokens,
                safety_margin=128,
            )
        )

    _sync(device)
    pass2_start = time.perf_counter()
    pass2_gen_kwargs = dict(gen_kwargs or {})
    pass2_max_new_tokens = int(pass2_gen_kwargs.pop("max_new_tokens", 50))
    predictions = generate_batch(
        model_pass2,
        tokenizer,
        pass2_prompts,
        visuals,
        device,
        vis_dim=vis_dim,
        max_new_tokens=pass2_max_new_tokens,
        microbatch=pass_batch_size,
        **pass2_gen_kwargs,
    )
    _sync(device)
    pass2_time = time.perf_counter() - pass2_start
    total_time = time.perf_counter() - total_start

    pass2_prompt_tokens = sum(
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in pass2_prompts
    )
    pass2_output_tokens = sum(
        len(tokenizer(text, add_special_tokens=False)["input_ids"])
        for text in predictions
    )
    peak_allocated = peak_reserved = 0.0
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
    info = {
        "chapters": len(video_samples),
        "pass1_time": pass1_time,
        "csse_time": csse_time,
        "pass2_time": pass2_time,
        "total_time": total_time,
        "pass1_batches": pass1_info["batches"],
        "pass2_batches": math.ceil(len(video_samples) / max(1, pass_batch_size)),
        "input_tokens": pass1_info["prompt_tokens"] + pass2_prompt_tokens,
        "output_tokens": pass1_info["output_tokens"] + pass2_output_tokens,
        "reference_generations": retrieval_system.stats["reference_generations"],
        "teacher_forced_forwards": retrieval_system.stats[
            "teacher_forced_forwards"
        ],
        "candidate_variants": retrieval_system.stats["candidate_variants"],
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "selected_contexts": selected_by_chapter,
        "support_details": support_details,
    }
    return [postprocess_title(p) for p in predictions], info

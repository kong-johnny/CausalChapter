# -*- coding: utf-8 -*-
"""Batched Pass-1 inference."""
from ..config.fewshot import FEWSHOT_EXAMPLES
from ..data.prompt_builder import build_prompt_parts
from ..data.vis_features import load_vis_seq
from ..utils.text_utils import build_prompt_with_truncation
from .generate import generate_batch


def generate_pass1_titles_for_video(
    model_pass1,
    tokenizer,
    example_id,
    video_samples,
    vis_index,
    vis_dim=512,
    vis_token_count=4,
    pass1_seg_tokens=6000,
    input_max_tokens=8192,
    use_fewshot=True,
    fewshot_num=3,
    device="cuda",
    gen_kwargs=None,
    batch_size=2,
    return_info=False,
):
    del pass1_seg_tokens  # token budget is enforced on the complete prompt
    examples = FEWSHOT_EXAMPLES[:fewshot_num] if use_fewshot else None
    prompts = []
    visuals = []
    for sample in video_samples:
        head, middle, tail = build_prompt_parts(
            segment_text=sample["segment_text"],
            context_subheadings=None,
            fewshot_examples=examples,
            num_query_tokens=vis_token_count,
            tokenizer=tokenizer,
        )
        prompts.append(
            build_prompt_with_truncation(
                tokenizer,
                head,
                middle,
                tail,
                max_total_tokens=input_max_tokens,
                safety_margin=128,
            )
        )
        visuals.append(
            load_vis_seq(
                vis_index,
                example_id,
                sample["start_idx"],
                sample["end_idx"],
                vis_dim=vis_dim,
                start_time=sample.get("start_time"),
                end_time=sample.get("end_time"),
            )
        )
    pass1_gen_kwargs = dict(gen_kwargs or {})
    pass1_max_new_tokens = int(pass1_gen_kwargs.pop("max_new_tokens", 50))
    predictions = generate_batch(
        model_pass1,
        tokenizer,
        prompts,
        visuals,
        device,
        vis_dim=vis_dim,
        max_new_tokens=pass1_max_new_tokens,
        microbatch=batch_size,
        **pass1_gen_kwargs,
    )
    if not return_info:
        return predictions
    return predictions, {
        "prompt_tokens": sum(
            len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            for prompt in prompts
        ),
        "output_tokens": sum(
            len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for text in predictions
        ),
        "batches": (len(prompts) + max(1, batch_size) - 1) // max(1, batch_size),
    }

# -*- coding: utf-8 -*-
"""Prompt construction shared by training and inference."""
from ..utils.utils import sharegpt_to_openai


SYSTEM_PROMPT = (
    "You are a helpful assistant generating lecture subheadings. "
    "Given the video segment transcript, generate a concise and informative "
    "subheading that captures the main topic. "
    "Only output the subheading, nothing else."
)


def build_vis_placeholder(num_query_tokens: int) -> str:
    if num_query_tokens <= 0:
        return ""
    return "".join(f"<VIS_{i}>" for i in range(num_query_tokens))


def _take_contexts(context_subheadings, max_context):
    contexts = list(context_subheadings or [])
    if max_context is None:
        return contexts
    return contexts[-max_context:]


def _build_user_prefix(
    context_subheadings=None,
    fewshot_examples=None,
    max_context=3,
    num_query_tokens=0,
    pass1_self_title=None,
):
    context_titles = _take_contexts(context_subheadings, max_context)
    vis_ph = build_vis_placeholder(num_query_tokens)
    user_prefix = f"Visual context: {vis_ph}\n\n" if vis_ph else ""

    if fewshot_examples:
        user_prefix += "Examples:\n"
        for ex in fewshot_examples:
            user_prefix += (
                f"Segment:\n{ex['segment']}\n"
                f"Subheading: {ex['title']}\n\n"
            )

    if context_titles:
        user_prefix += "Related subheadings (same lecture):\n"
        user_prefix += "\n".join(f"- {title}" for title in context_titles)
        user_prefix += "\n\n"

    if pass1_self_title:
        user_prefix += (
            "Initial draft for current segment:\n"
            f"{pass1_self_title}\n\n"
        )
    return user_prefix


def build_prompt_parts(
    segment_text,
    context_subheadings=None,
    fewshot_examples=None,
    max_context=3,
    num_query_tokens=0,
    pass1_self_title=None,
    tokenizer=None,
):
    """Return head/middle/tail for sandwich truncation.

    `max_context=None` keeps the complete compact candidate set. This is used
    only during CSSE scoring. Pass 2 normally receives the final Top-K set.
    """
    user_prefix = _build_user_prefix(
        context_subheadings=context_subheadings,
        fewshot_examples=fewshot_examples,
        max_context=max_context,
        num_query_tokens=num_query_tokens,
        pass1_self_title=pass1_self_title,
    )
    refine_instruction = ""
    if pass1_self_title:
        refine_instruction = (
            "\nRefine the initial draft into a more accurate subheading "
            "based on the segment."
        )

    if tokenizer is None:
        raise ValueError(
            "tokenizer is required: silently falling back to a Qwen template "
            "produces invalid prompts for LLaMA."
        )

    sentinel = "__SEGMENT_TEXT_PLACEHOLDER_DO_NOT_GENERATE__"
    user_content = f"{user_prefix}Segment:\n{sentinel}{refine_instruction}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    templated = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if sentinel not in templated:
        raise RuntimeError("Sentinel not found in tokenizer chat template output.")
    head, tail = templated.split(sentinel, 1)
    return head, str(segment_text), tail


def build_inference_prompt(tokenizer, messages):
    msgs = [m for m in messages if m["from"] in ["system", "human"]]
    return tokenizer.apply_chat_template(
        sharegpt_to_openai(msgs),
        tokenize=False,
        add_generation_prompt=True,
    )

# -*- coding: utf-8 -*-
"""Construct Pass-2 training samples with paper-aligned KL-CSSE contexts."""
from tqdm import tqdm

from ..config.fewshot import FEWSHOT_EXAMPLES
from ..data.segment_builder import group_samples_by_video
from ..data.vis_features import load_vis_seq
from ..retrieval.kl_support_retriever import KLSupportRetriever
from ..utils.text_utils import postprocess_title
from .pass1_inference import generate_pass1_titles_for_video


def build_pass2_causal_samples(
    samples,
    model_pass1,
    tokenizer,
    vis_index,
    retrieval_system=None,
    vis_dim=512,
    vis_token_count=4,
    causal_topk=5,
    candidate_size=10,
    pass1_seg_tokens=6000,
    input_max_tokens=8192,
    use_fewshot=True,
    fewshot_num=3,
    device="cuda",
    gen_kwargs=None,
    pass_batch_size=2,
    variant_microbatch=2,
):
    if not isinstance(retrieval_system, KLSupportRetriever):
        raise TypeError(
            "Pass-2 construction for the paper configuration requires "
            "KLSupportRetriever instantiated with the multimodal wrapper."
        )
    by_eid = group_samples_by_video(samples)
    output = []
    fewshot = FEWSHOT_EXAMPLES[:fewshot_num] if use_fewshot else None
    for eid, video_samples in tqdm(by_eid.items(), desc="Build KL-CSSE Pass2 samples"):
        pass1 = generate_pass1_titles_for_video(
            model_pass1=model_pass1,
            tokenizer=tokenizer,
            example_id=eid,
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
        )
        retrieval_segments = [
            {
                "text": sample["segment_text"],
                "title": pass1[idx],
                "index": idx,
                "time_span": [sample["start_time"], sample["end_time"]],
            }
            for idx, sample in enumerate(video_samples)
        ]
        retrieval_system.ensure_built(retrieval_segments, video_key=eid)
        for idx, sample in enumerate(video_samples):
            visual = load_vis_seq(
                vis_index,
                eid,
                sample["start_idx"],
                sample["end_idx"],
                vis_dim=vis_dim,
                start_time=sample.get("start_time"),
                end_time=sample.get("end_time"),
            )
            selected, _ = retrieval_system.select_contexts(
                retrieval_segments[idx],
                retrieval_segments,
                idx,
                visual,
                candidate_size=candidate_size,
                topk=causal_topk,
                fewshot_examples=fewshot,
                gen_kwargs=gen_kwargs,
            )
            output.append(
                {
                    **sample,
                    "causal_context_titles": [
                        postprocess_title(pass1[j]) for j in selected
                    ],
                    "pass1_title": pass1[idx],
                    "source": sample.get("source", "gt"),
                }
            )
    return output

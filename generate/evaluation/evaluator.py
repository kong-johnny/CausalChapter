# -*- coding: utf-8 -*-
"""eval_model"""
import json
import os
import numpy as np
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

from ..models.SODA.soda import SODA
from ..utils.text_utils import text_jaccard, postprocess_title
from ..utils.match_utils import hungarian_match
from ..inference.causal_selection import causal_selection_inference


def eval_model(
    model_pass1,
    model_pass2,
    tokenizer,
    eval_data,
    vis_index,
    retrieval_system=None,
    vis_dim=512,
    vis_token_count=4,
    topk=3,
    pass1_seg_tokens=6000,
    pass2_seg_tokens=4500,
    input_max_tokens=8192,
    use_fewshot=True,
    fewshot_num=3,
    device="cuda",
    gen_kwargs=None,
    save_dir=None,         # 若指定，保存每段输入输出到NAS
    boundary_type="",      # "gt" / "pred"，用于文件名区分
    epoch=None,            # epoch数，用于文件名
):
    all_gts = {}
    all_preds = {}
    soda_gts = {}
    soda_preds = {}
    detail_records = []    # 收集每段输入输出详情

    total_videos = 0
    total_preds = 0
    empty_preds = 0
    matched_pairs = 0

    for item in tqdm(eval_data, desc="Evaluating"):
        eid = item["example_id"]
        segments = [
            {"segment_text": seg} for seg in item["segments"]
        ]
        spans = item["spans"]
        times = item["times"]
        gt_titles = [x for x in item["gt_titles"]]

        if len(segments) == 0 or len(gt_titles) == 0:
            continue

        preds, _info = causal_selection_inference(
            model_pass1=model_pass1,
            model_pass2=model_pass2,
            tokenizer=tokenizer,
            example_id=eid,
            segments=segments,
            spans=spans,
            vis_index=vis_index,
            times=times,
            retrieval_system=retrieval_system,
            vis_dim=vis_dim,
            vis_token_count=vis_token_count,
            candidate_size=10,
            topk=topk,
            pass1_seg_tokens=pass1_seg_tokens,
            pass2_seg_tokens=pass2_seg_tokens,
            input_max_tokens=input_max_tokens,
            use_fewshot=use_fewshot,
            fewshot_num=fewshot_num,
            device=device,
            gen_kwargs=gen_kwargs,
        )
        preds = [postprocess_title(x) for x in preds]

        total_videos += 1
        total_preds += len(preds)
        empty_preds += sum(1 for x in preds if len(x) == 0)

        sim = np.zeros((len(preds), len(gt_titles)), dtype=np.float32)
        for i, p in enumerate(preds):
            for j, g in enumerate(gt_titles):
                sim[i, j] = text_jaccard(p, g)

        matches = hungarian_match(sim)
        matched_pairs += len(matches)

        for pi, gj in matches:
            key = f"{eid}_{pi}"
            all_preds[key] = [preds[pi]]
            all_gts[key] = [gt_titles[gj]]

        # 收集详细记录（输入输出对）
        detail_records.append({
            "example_id": eid,
            "boundary_type": boundary_type,
            "segments": item["segments"],
            "predictions": preds,
            "gt_titles": gt_titles,
            "times": times,
            "matches": [(int(pi), int(gj)) for pi, gj in matches],
            "epoch": epoch,
            "inference_info": _info,
        })

        pred_events = []
        for i, (st, ed) in enumerate(times):
            sent = preds[i] if i < len(preds) else ""
            pred_events.append({
                "sentence": sent,
                "timestamp": [float(st), float(ed)],
            })
        soda_preds[eid] = pred_events

        if "gt_events" in item:
            soda_gts[eid] = {
                "sentences": [x["sentence"] for x in item["gt_events"]],
                "timestamps": [[float(x["start"]), float(x["end"])] for x in item["gt_events"]],
            }
        else:
            soda_gts[eid] = {
                "sentences": item["gt_titles"],
                "timestamps": [[float(st), float(ed)] for (st, ed) in item["times"]],
            }

    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0

    cider_score, _ = Cider().compute_score(all_gts, all_preds)

    try:
        meteor_result = Meteor().compute_score(all_gts, all_preds)
        if isinstance(meteor_result, tuple):
            meteor_score = meteor_result[0]
            if isinstance(meteor_score, (list, np.ndarray)):
                meteor_score = float(np.mean(meteor_score))
            else:
                meteor_score = float(meteor_score)
        else:
            meteor_score = float(meteor_result)
    except Exception as e:
        print(f"[WARN] METEOR failed: {e}")
        meteor_score = 0.0

    evaluator = SODA.build(
        preds=soda_preds,
        gts=[soda_gts],
        gt_vids=sorted(soda_gts.keys()),
        soda_type="c",
        tious=[0.0],
        scorer="Meteor",
        verbose=False,
    )
    soda_result = evaluator.evaluate()
    soda_score = float(soda_result["Meteor"][2])

    print(f"[EVAL] total_videos={total_videos}")
    print(f"[EVAL] total_preds={total_preds}, empty_preds={empty_preds}")
    print(f"[EVAL] matched_pairs={matched_pairs}")

    # 保存详细输入输出到NAS
    if save_dir and detail_records:
        epoch_str = f"_epoch{epoch}" if epoch is not None else ""
        btype_str = f"_{boundary_type}" if boundary_type else ""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"eval_details{btype_str}{epoch_str}.jsonl")
        with open(save_path, "w", encoding="utf-8") as f:
            for rec in detail_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[SAVE] eval details -> {save_path} ({len(detail_records)} videos)")

    return cider_score * 100.0, meteor_score * 100.0, soda_score * 100.0
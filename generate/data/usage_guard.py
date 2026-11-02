"""Small, dependency-free guards for GT-free development searches."""
from pathlib import Path


def validate_evaluation_usage(scope, protocol, caption_matching, forbid_gt, dev_only, text_path, gt_path):
    if forbid_gt and (scope != "pred" or protocol != "strict"):
        raise ValueError("GT-free runs require pred scope and strict all-predicted-segment construction")
    if caption_matching not in ("temporal", "lexical"):
        raise ValueError("CC_CAPTION_MATCHING must be temporal or lexical")
    if dev_only and (Path(text_path).name != "dev.jsonl" or Path(gt_path).name != "dev_subheading_abstract.jsonl"):
        raise ValueError("Parameter search must use the original development split")

#!/usr/bin/env python3
"""Bounded existing-weight/decoding search on dev, predicted segments only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", nargs="+", required=True)
    parser.add_argument("--boundaries", required=True)
    parser.add_argument("--chapter-times")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--visual-cache", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=8, help="First stage dev subset only; 0 means full dev")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    settings = [
        {"beams": 3, "repetition": 1.005, "topk": 3},
        {"beams": 5, "repetition": 1.005, "topk": 3},
        {"beams": 3, "repetition": 1.005, "topk": 5},
        {"beams": 3, "repetition": 1.05, "topk": 3},
    ]
    plan = {**vars(args), "split": "dev", "gt_segmentation": "forbidden",
            "caption_matching": "lexical, fixed before search to audit original CIDEr matching",
            "temporal_evaluation": "all predictions and true reference timestamps",
            "dev_sha256": hashlib.sha256((Path(args.data_dir) / "dev.jsonl").read_bytes()).hexdigest(),
            "boundary_sha256": hashlib.sha256(Path(args.boundaries).read_bytes()).hexdigest(),
            "settings": settings}
    plan_path = output / "search_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Refusing to mix different plans in one search directory")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    results = []
    for generator_index, checkpoint in enumerate(args.generator):
        meta = json.loads((Path(checkpoint) / "multimodal_meta.json").read_text())
        for setting_index, setting in enumerate(settings):
            run = output / f"g{generator_index}_s{setting_index}"
            run.mkdir(exist_ok=True)
            summary = run / "eval_summary.json"
            if not summary.exists():
                environment = {**os.environ, "USE_TF": "0", "USE_FLAX": "0", "OMP_NUM_THREADS": "1",
                               "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                               "CC_SEARCH_DEV_ONLY": "1", "CC_FORBID_GT_SEGMENTS": "1",
                               "CC_EVAL_PROTOCOL": "strict", "CC_CAPTION_MATCHING": "lexical",
                               "CC_EVAL_SCOPE": "pred", "CC_EVAL_LIMIT": str(args.limit),
                               "CC_EVAL_TEXT_JSONL": str(Path(args.data_dir) / "dev.jsonl"),
                               "CC_EVAL_GT_JSONL": str(Path(args.data_dir) / "dev_subheading_abstract.jsonl"),
                               "CC_BASE_MODEL": args.base_model, "CC_CHECKPOINT_DIR": checkpoint,
                               "CC_FUSION_MODE": meta["fusion_mode"], "CC_NUM_QUERY_TOKENS": str(meta["num_query_tokens"]),
                               "CC_VIS_DIM": "512", "CC_VIS_CACHE_DIR": args.visual_cache,
                               "CC_TEST_PRED_BOUNDARY": args.boundaries, "CC_EVAL_SAVE_DIR": str(run),
                               "CC_SEMANTIC_BACKEND": "hashing", "CC_SEED": "42", "CC_CAUSAL_MODE": "light",
                               "CC_CAUSAL_TOPK": str(setting["topk"]), "CC_RETRIEVAL_NEAR_SIZE": "3",
                               "CC_RETRIEVAL_SEMANTIC_SIZE": "10", "CC_RETRIEVAL_INTERVENTION_TOKENS": "16",
                               "CC_RETRIEVAL_VARIANT_MICROBATCH": "2", "CC_USE_FEWSHOT": "0",
                               "CC_GEN_NUM_BEAMS": str(setting["beams"]), "CC_GEN_REPETITION_PENALTY": str(setting["repetition"]),
                               "CC_GEN_NO_REPEAT_NGRAM_SIZE": "3"}
                # No stale test timeline may enter a dev search through the shell.
                environment.pop("CC_CHAPTER_TIMES", None)
                if args.chapter_times:
                    environment["CC_CHAPTER_TIMES"] = args.chapter_times
                print(f"START generator={generator_index} setting={setting_index}: {setting}", flush=True)
                with (run / "run.log").open("w") as log:
                    process = subprocess.Popen([sys.executable, "-u", "-m", "generate.eval_env"],
                                               cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    (run / "run.pid").write_text(str(process.pid) + "\n")
                    if process.wait() != 0:
                        raise RuntimeError(f"Candidate failed; inspect {run / 'run.log'} before any restart")
            result = {"generator": checkpoint, "settings": setting, "summary": json.loads(summary.read_text())}
            results.append(result)
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    print("DEV_SWEEP_COMPLETE; no GT segments used; no test split loaded", flush=True)


if __name__ == "__main__":
    main()

"""Inference from the released LCDS weights; no annotation file is required."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .decoding import decode_probabilities
from .train_lcds import LongformerLCDS, VideoInputs, collect_scores, read_rows
from .utils import int_to_label, save_boundaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/lcds_longformer_reconstruction")
    parser.add_argument("--text", required=True, help="JSONL with example_id, text, stet; labels are not used")
    parser.add_argument("--visual-cache", required=True, help="Directory of per-utterance 512D .npy vectors")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint / "best_config.json").read_text())
    selection = json.loads((checkpoint / "dev_decoding_tuning.json").read_text())
    if selection["selection_split"] != "dev" or selection["dev_sha256"] != config["dev_sha256"]:
        raise ValueError("Frozen decoding and checkpoint development split differ")
    decoding = selection["selected"]["decoding"]
    anchor = decoding["boundary_anchor"]
    if anchor not in ("start", "middle", "end"):
        raise ValueError("Invalid timestamp anchor")
    torch.manual_seed(config["seed"])
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    model = LongformerLCDS(None, config["hidden_size"], config["window_size"],
                           config["freeze_encoder"], encoder_config=config["encoder_config"],
                           visual_radius=config.get("visual_radius"))
    model.load_state_dict(torch.load(checkpoint / "best.pt", map_location="cpu", weights_only=True), strict=True)
    model.to(device)
    rows = read_rows(args.text)
    inputs = VideoInputs(rows, tokenizer, args.visual_cache, config["max_tokens"])
    scores = collect_scores(model, inputs, device)
    labels = {eid: decode_probabilities(values, decoding["threshold"], decoding["min_gap"]) for eid, values in scores.items()}
    timelines = []
    for row in rows:
        indices = [i for i, value in enumerate(labels[row["example_id"]][:-1]) if value]
        times = []
        for index in indices:
            start, end = map(float, row["stet"][index])
            times.append({"start": start, "middle": (start + end) / 2, "end": end}[anchor])
        endpoints = [0.0, *times, float(row["stet"][-1][1])]
        chapters = [[start, end] for start, end in zip(endpoints, endpoints[1:])]
        if any(end <= start for start, end in chapters):
            raise ValueError(f"Nonpositive chapter interval: {row['example_id']}")
        timelines.append({"example_id": row["example_id"], "boundary_anchor": anchor,
                          "boundary_indices": indices, "boundary_seconds": times, "chapters": chapters})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_boundaries({eid: {"predictions": int_to_label(values)} for eid, values in labels.items()}, str(output / "boundaries.jsonl"))
    (output / "chapter_times.jsonl").write_text("".join(json.dumps(row) + "\n" for row in timelines))
    (output / "scores.json").write_text(json.dumps(scores) + "\n")
    print(f"LCDS_INFERENCE_COMPLETE: {len(rows)} videos; annotations were not loaded", flush=True)


if __name__ == "__main__":
    main()

"""Generation inputs contain predictions only; references supply loss targets."""
import math


def validate_prediction_artifacts(texts, boundaries, timelines):
    """Fail before model loading if predicted chapter artifacts are malformed."""
    def index(rows):
        result = {}
        for row in rows:
            eid = row["example_id"]
            if isinstance(eid, list):
                if not eid or any(value != eid[0] for value in eid):
                    raise ValueError("Inconsistent repeated ID")
                if len(eid) not in (1, len(row.get("predictions", []))):
                    raise ValueError("Repeated ID length mismatch")
                eid = eid[0]
            if eid in result:
                raise ValueError("Duplicate prediction/timeline ID")
            result[eid] = row
        return result
    text_map, boundary_map, time_map = map(index, (texts, boundaries, timelines))
    if not (text_map.keys() == boundary_map.keys() == time_map.keys()):
        raise ValueError("Prediction/timeline IDs differ from split")
    for eid, text in text_map.items():
        labels = boundary_map[eid]["predictions"]
        if len(labels) != len(text["text"]) or not labels:
            raise ValueError("Boundary length mismatch or empty transcript")
        if any(value not in (0, 1, "O", "B-EOP") for value in labels):
            raise ValueError("Invalid boundary label")
        indices = [i for i, value in enumerate(labels[:-1]) if value in (1, "B-EOP")]
        timeline = time_map[eid]
        if timeline["boundary_indices"] != indices or len(timeline["chapters"]) != len(indices) + 1:
            raise ValueError("Timeline does not describe predicted boundaries")
        previous_end = 0.0
        for start, end in timeline["chapters"]:
            if not (math.isfinite(start) and math.isfinite(end)) or start != previous_end or end <= start:
                raise ValueError("Noncontiguous/nonpositive timeline")
            previous_end = end
        if abs(previous_end - float(text["stet"][-1][1])) > 1e-5:
            raise ValueError("Timeline does not cover the video")


def prediction_samples(items):
    samples = []
    for item in items:
        segments, spans, times = item["segments"], item["spans"], item["times"]
        if not (len(segments) == len(spans) == len(times)):
            raise ValueError("Prediction text/span/time lengths differ")
        for text, (start_idx, end_idx), (start, end) in zip(segments, spans, times):
            if end <= start or end_idx <= start_idx:
                raise ValueError("Nonpositive predicted segment")
            samples.append({"example_id": item["example_id"], "segment_text": text,
                            "start_idx": start_idx, "end_idx": end_idx,
                            "start_time": start, "end_time": end, "source": "pred"})
    return samples


def supervised_predictions(samples, items, minimum_iou=0.3):
    """Select loss-bearing TRAIN predictions; never call this for inference.

    CSSE generation receives prediction_samples() BEFORE this function is
    applied, including unmatched chapters. This function only attaches teacher
    labels for gradient training and never changes segment text or boundaries.
    """
    references = {item["example_id"]: item["gt_events"] for item in items}
    labeled = []
    for sample in samples:
        best = None
        best_iou = 0.0
        for event in references[sample["example_id"]]:
            start, end = sample["start_time"], sample["end_time"]
            intersection = max(0.0, min(end, event["end"]) - max(start, event["start"]))
            union = max(end, event["end"]) - min(start, event["start"])
            iou = intersection / max(union, 1e-8)
            if iou > best_iou:
                best, best_iou = event, iou
        if best is not None and best_iou >= minimum_iou:
            labeled.append({**sample, "target_subheading": best["sentence"], "training_target_iou": best_iou})
    if not labeled:
        raise ValueError("No training predictions have valid reference loss targets")
    return labeled

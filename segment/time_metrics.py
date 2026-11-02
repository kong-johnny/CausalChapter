"""Explicit, one-to-one time-domain boundary metrics (standard library only)."""
from __future__ import annotations


def match_times(predicted, reference, tolerance):
    """Maximize matched pairs, then minimize their summed absolute error."""
    predicted, reference = sorted(predicted), sorted(reference)
    table = [[(0, 0.0) for _ in range(len(reference) + 1)] for _ in range(len(predicted) + 1)]
    for i, value in enumerate(predicted, 1):
        for j, target in enumerate(reference, 1):
            candidates = [table[i - 1][j], table[i][j - 1]]
            error = abs(value - target)
            if error <= tolerance:
                count, total = table[i - 1][j - 1]
                candidates.append((count + 1, total + error))
            table[i][j] = max(candidates, key=lambda item: (item[0], -item[1]))
    return table[-1][-1]


def prf(tp, predicted_count, reference_count):
    precision = tp / max(predicted_count, 1)
    recall = tp / max(reference_count, 1)
    return {"P": 100 * precision, "R": 100 * recall, "F1": 200 * tp / max(predicted_count + reference_count, 1)}


def evaluate_internal_boundaries(rows, predictions):
    if set(predictions) != {row["example_id"] for row in rows}:
        raise ValueError("Prediction video IDs must exactly match the evaluation split")
    exact_tp = predicted_count = reference_count = 0
    time_matches = {seconds: [0, 0.0] for seconds in (5, 10, 30)}
    count_errors = []
    for row in rows:
        predicted = predictions[row["example_id"]]
        reference = row["labels"]
        if len(predicted) != len(reference) or len(reference) != len(row["stet"]):
            raise ValueError(f"Length mismatch for {row['example_id']}")
        # Both metrics exclude the terminal endpoint, including single-chapter videos.
        p_indices = [index for index, value in enumerate(predicted[:-1]) if value]
        r_indices = [index for index, value in enumerate(reference[:-1]) if value]
        predicted_count += len(p_indices)
        reference_count += len(r_indices)
        exact_tp += len(set(p_indices) & set(r_indices))
        count_errors.append(abs(len(p_indices) - len(r_indices)))
        p_times = [float(row["stet"][index][1]) for index in p_indices]
        # Evaluate localization against annotated chapter times when available.
        r_times = [float(value) for value in row.get("topic_end_seconds", [])[:-1]]
        if len(r_times) != len(r_indices):
            r_times = [float(row["stet"][index][1]) for index in r_indices]
        for tolerance in time_matches:
            count, error = match_times(p_times, r_times, tolerance)
            time_matches[tolerance][0] += count
            time_matches[tolerance][1] += error
    output = {
        "videos": len(rows),
        "internal_boundary_P": prf(exact_tp, predicted_count, reference_count)["P"],
        "internal_boundary_R": prf(exact_tp, predicted_count, reference_count)["R"],
        "internal_boundary_F1": prf(exact_tp, predicted_count, reference_count)["F1"],
        "predicted_boundaries": predicted_count,
        "reference_boundaries": reference_count,
        "boundary_count_MAE": sum(count_errors) / max(len(rows), 1),
    }
    for tolerance, (count, error) in time_matches.items():
        output.update({f"{key}@{tolerance}s": value for key, value in prf(count, predicted_count, reference_count).items()})
        output[f"matched_MAE@{tolerance}s"] = error / count if count else None
    return output

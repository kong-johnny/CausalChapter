import unittest

from segment.time_metrics import evaluate_internal_boundaries, match_times


class TimeMetricsTest(unittest.TestCase):
    def test_matching_maximizes_one_to_one_hits(self):
        count, error = match_times([5, 10], [0, 6], 5)
        self.assertEqual(count, 2)
        self.assertEqual(error, 9)

    def test_terminal_endpoint_is_not_counted_as_prediction(self):
        row = {"example_id": "v", "labels": [0, 1, 1], "stet": [[0, 10], [10, 20], [20, 30]], "topic_end_seconds": [20, 30]}
        result = evaluate_internal_boundaries([row], {"v": [0, 1, 0]})
        self.assertEqual(result["internal_boundary_F1"], 100)
        self.assertEqual(result["F1@5s"], 100)
        self.assertEqual(result["reference_boundaries"], 1)

    def test_missing_video_is_an_error(self):
        with self.assertRaises(ValueError):
            evaluate_internal_boundaries([{"example_id": "v"}], {})

    def test_anchor_is_explicit_and_does_not_change_exact_label_metric(self):
        row = {"example_id": "v", "labels": [1, 0], "stet": [[0, 20], [20, 30]], "topic_end_seconds": [10, 30]}
        end = evaluate_internal_boundaries([row], {"v": [1, 0]})
        middle = evaluate_internal_boundaries([row], {"v": [1, 0]}, "middle")
        self.assertEqual(end["F1@5s"], 0)
        self.assertEqual(middle["F1@5s"], 100)
        self.assertEqual(end["internal_boundary_F1"], middle["internal_boundary_F1"])

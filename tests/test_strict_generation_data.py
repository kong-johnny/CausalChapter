import json
from pathlib import Path
import tempfile
import unittest

from generate.data.eval_data_builder import build_strict_pred_eval_data


class StrictGenerationDataTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.paths = [root / name for name in ("text.jsonl", "gt.jsonl", "pred.jsonl", "times.jsonl")]
        rows = [
            {"example_id": "v", "text": ["one", "two", "three", "four"], "stet": [[0, 10], [10, 20], [20, 30], [30, 40]]},
            {"example_id": "v", "subheading_abstract": [{"text": "one two", "subheading": "first", "start": 0, "end": 20}, {"text": "three four", "subheading": "second", "start": 20, "end": 40}]},
            {"example_id": ["v"], "predictions": [1, 1, 0, 0]},
            {"example_id": "v", "boundary_indices": [0, 1], "chapters": [[0, 5], [5, 15], [15, 40]]},
        ]
        for path, row in zip(self.paths, rows):
            path.write_text(json.dumps(row) + "\n")

    def test_preserves_unmatched_predictions_and_real_reference_times(self):
        data = build_strict_pred_eval_data(*map(str, self.paths))[0]
        self.assertEqual(len(data["segments"]), 3)
        self.assertEqual(len(data["gt_titles"]), 2)
        self.assertEqual(data["times"], [(0, 5), (5, 15), (15, 40)])
        self.assertEqual(data["gt_events"][0]["end"], 20)
        self.assertEqual(data["gt_events"][1]["start"], 20)
        self.assertEqual(data["segments"], ["one", "two", "three four"])

    def test_rejects_different_prediction_id(self):
        self.paths[2].write_text(json.dumps({"example_id": "other", "predictions": [0]}) + "\n")
        with self.assertRaises(ValueError):
            build_strict_pred_eval_data(*map(str, self.paths))

    def test_rejects_timeline_for_different_boundaries(self):
        row = json.loads(self.paths[3].read_text())
        row["boundary_indices"] = [0, 2]
        self.paths[3].write_text(json.dumps(row) + "\n")
        with self.assertRaises(ValueError):
            build_strict_pred_eval_data(*map(str, self.paths))

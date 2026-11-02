import unittest

from generate.data.clean_training import prediction_samples, supervised_predictions, validate_prediction_artifacts


class CleanGeneratorInputTest(unittest.TestCase):
    def item(self):
        return {"example_id": "train", "segments": ["first prediction", "unmatched prediction"],
                "spans": [(0, 2), (2, 4)], "times": [(0, 2), (2, 4)],
                "gt_titles": ["secret target"],
                "gt_events": [{"start": 0, "end": 2, "sentence": "secret target"}]}

    def test_all_predictions_and_no_reference_fields(self):
        samples = prediction_samples([self.item()])
        self.assertEqual(len(samples), 2)
        self.assertTrue(all("target_subheading" not in sample for sample in samples))
        self.assertNotIn("secret target", str(samples))

    def test_reference_change_cannot_change_generation_input(self):
        item = self.item()
        original = prediction_samples([item])
        item["gt_titles"] = ["different"]
        item["gt_events"] = []
        self.assertEqual(original, prediction_samples([item]))

    def test_loss_labels_do_not_mutate_generation_samples(self):
        item = self.item()
        raw = prediction_samples([item])
        labeled = supervised_predictions(raw, [item])
        self.assertEqual(len(raw), 2)
        self.assertEqual(len(labeled), 1)
        self.assertEqual(labeled[0]["segment_text"], raw[0]["segment_text"])
        self.assertEqual(labeled[0]["target_subheading"], "secret target")
        self.assertNotIn("target_subheading", raw[0])

    def test_bad_lengths_fail(self):
        item = self.item()
        item["spans"] = []
        with self.assertRaises(ValueError):
            prediction_samples([item])

    def test_artifacts_use_predictions_not_text_labels(self):
        texts = [{"example_id": "train", "text": ["a", "b"], "stet": [[0, 1], [1, 2]],
                  "labels": [0, 0]}]
        boundaries = [{"example_id": "train", "predictions": [1, 0]}]
        times = [{"example_id": "train", "boundary_indices": [0], "chapters": [[0, 0.5], [0.5, 2]]}]
        validate_prediction_artifacts(texts, boundaries, times)
        times[0]["boundary_indices"] = []
        with self.assertRaisesRegex(ValueError, "predicted boundaries"):
            validate_prediction_artifacts(texts, boundaries, times)

    def test_artifacts_reject_uncovered_time(self):
        with self.assertRaisesRegex(ValueError, "cover the video"):
            validate_prediction_artifacts(
                [{"example_id": "train", "text": ["a"], "stet": [[0, 2]]}],
                [{"example_id": "train", "predictions": [0]}],
                [{"example_id": "train", "boundary_indices": [], "chapters": [[0, 1]]}])


if __name__ == "__main__":
    unittest.main()

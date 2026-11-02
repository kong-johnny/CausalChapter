import json
from pathlib import Path
import tempfile
import unittest

from generate.train_clean import audit, parser


class CleanGeneratorAuditTest(unittest.TestCase):
    def fixture(self, root):
        # Synthetic metadata/weights for dependency-free audit unit tests only;
        # these are never loaded by a model or claimed to be real artifacts.
        base = root / "base"
        base.mkdir()
        (base / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
        (base / "model.safetensors").write_bytes(b"unit-test-only")
        for split in ("train", "dev"):
            row = {"example_id": split, "text": [f"{split} words"], "stet": [[0, 2]]}
            (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
            (root / f"{split}_subheading_abstract.jsonl").write_text(json.dumps({"example_id": split}) + "\n")
            (root / f"{split}_boundaries.jsonl").write_text(json.dumps({"example_id": split, "predictions": [0]}) + "\n")
            (root / f"{split}_times.jsonl").write_text(json.dumps({"example_id": split,
                "boundary_indices": [], "chapters": [[0, 2]]}) + "\n")
        return parser().parse_args([
            "--data-dir", str(root), "--base-model", str(base), "--visual-cache", str(root),
            "--train-boundaries", str(root / "train_boundaries.jsonl"),
            "--train-times", str(root / "train_times.jsonl"),
            "--dev-boundaries", str(root / "dev_boundaries.jsonl"),
            "--dev-times", str(root / "dev_times.jsonl"), "--output", str(root / "output")])

    def test_valid_preflight_without_ml(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            _, receipt = audit(args)
            self.assertEqual(receipt["training_videos"], 1)
            self.assertFalse(receipt["dev_used_for_gradient_training"])
            self.assertEqual(len(receipt["input_sha256"]), 8)

    def test_swapped_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args.dev_boundaries = args.train_boundaries
            with self.assertRaisesRegex(ValueError, "ID set differs"):
                audit(args)

    def test_inherited_adapter_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            (Path(args.base_model) / "adapter_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "inherited adapter"):
                audit(args)


if __name__ == "__main__":
    unittest.main()

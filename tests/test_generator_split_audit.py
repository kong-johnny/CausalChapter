import json
from pathlib import Path
import tempfile
import unittest

from generate.data.split_audit import audited_train_rows


class GeneratorSplitAuditTest(unittest.TestCase):
    def run_audit(self, train, dev, annotations=None):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("train.jsonl", "annotations.jsonl", "dev.jsonl")]
            rows = (train, annotations if annotations is not None else train, dev)
            for path, values in zip(paths, rows):
                path.write_text("\n".join(json.dumps(value) for value in values))
            return audited_train_rows(*paths)

    def test_disjoint(self):
        train = [{"example_id": "train", "text": ["training words"]}]
        texts, annotations, receipt = self.run_audit(train, [{"example_id": "dev", "text": ["validation words"]}])
        self.assertEqual(texts, train)
        self.assertEqual(annotations, train)
        self.assertFalse(receipt["dev_used_for_gradient_training"])

    def test_augmented_id_overlap(self):
        with self.assertRaisesRegex(ValueError, "IDs overlap"):
            self.run_audit([{"example_id": "dev__predaug", "text": ["training words"]}],
                           [{"example_id": "dev", "text": ["validation words"]}])

    def test_renamed_duplicate(self):
        with self.assertRaisesRegex(ValueError, "transcripts overlap"):
            self.run_audit([{"example_id": "train", "text": ["Same, CONTENT!"]}],
                           [{"example_id": "dev", "text": ["same content"]}])

    def test_annotation_ids(self):
        with self.assertRaisesRegex(ValueError, "ID sets differ"):
            self.run_audit([{"example_id": "train", "text": ["training"]}],
                           [{"example_id": "dev", "text": ["validation"]}],
                           [{"example_id": "other"}])


if __name__ == "__main__":
    unittest.main()

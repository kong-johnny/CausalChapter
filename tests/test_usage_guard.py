import unittest
from generate.data.usage_guard import validate_evaluation_usage


class UsageGuardTest(unittest.TestCase):
    def validate(self, scope="pred", protocol="strict", text="dev.jsonl"):
        validate_evaluation_usage(scope, protocol, "lexical", True, True, text, "dev_subheading_abstract.jsonl")

    def test_predicted_dev_is_allowed(self):
        self.validate()

    def test_oracle_is_forbidden(self):
        for scope in ("gt", "both"):
            with self.assertRaises(ValueError):
                self.validate(scope=scope)

    def test_gt_filtered_legacy_is_forbidden(self):
        with self.assertRaises(ValueError):
            self.validate(protocol="legacy")

    def test_test_split_is_forbidden_for_search(self):
        with self.assertRaises(ValueError):
            self.validate(text="test.jsonl")

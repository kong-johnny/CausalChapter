import json
from pathlib import Path
import tempfile
import unittest

from generate.data.tokenizer_compat import tokenizer_compat_kwargs


class TokenizerCompatibilityTest(unittest.TestCase):
    def check_config(self, config, expected):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer_config.json"
            original = json.dumps(config)
            path.write_text(original)
            self.assertEqual(tokenizer_compat_kwargs(directory), expected)
            self.assertEqual(path.read_text(), original)

    def test_list_merges_without_rewriting(self):
        self.check_config({"extra_special_tokens": ["<a>", "<b>"],
                           "additional_special_tokens": ["<a>"]},
                          {"extra_special_tokens": {}, "additional_special_tokens": ["<a>", "<b>"]})

    def test_dictionary_unchanged(self):
        self.check_config({"extra_special_tokens": {"image_token": "<image>"}}, {})

    def test_missing_unchanged(self):
        self.check_config({}, {})


if __name__ == "__main__":
    unittest.main()

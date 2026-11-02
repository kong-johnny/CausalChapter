"""Read-only compatibility for list-valued HF 5 tokenizer metadata on HF 4."""
import json
from pathlib import Path


def tokenizer_compat_kwargs(source):
    config = json.loads((Path(source) / "tokenizer_config.json").read_text())
    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return {}
    if not all(isinstance(token, str) for token in extra):
        raise ValueError("List-valued extra_special_tokens must contain strings")
    # HF 4 expects extra_special_tokens to be a named-token dictionary. HF 5
    # also serializes a plain list; pass it as additional tokens without
    # rewriting the checkpoint or changing its existing added-token IDs.
    tokens = list(dict.fromkeys((config.get("additional_special_tokens") or []) + extra))
    return {"extra_special_tokens": {}, "additional_special_tokens": tokens}

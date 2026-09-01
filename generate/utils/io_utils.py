# -*- coding: utf-8 -*-
"""JSON / JSONL I/O utilities"""
import json


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_example_map(rows):
    return {x["example_id"]: x for x in rows if "example_id" in x}
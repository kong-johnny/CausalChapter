"""Dependency-free train/dev isolation checks for new generator experiments."""
import hashlib
import json
from pathlib import Path
import re


def canonical_id(row):
    value = row["example_id"]
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty string example_id")
    return value.removesuffix("__predaug")


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def transcript_digest(row):
    text = row["text"]
    if isinstance(text, list):
        text = " ".join(text)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected nonempty transcript")
    normalized = re.sub(r"\W+", " ", text.casefold()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def audited_train_rows(train_text, train_annotations, dev_text):
    texts, annotations, dev = map(load_rows, (train_text, train_annotations, dev_text))
    train_ids = {canonical_id(row) for row in texts}
    annotation_ids = {canonical_id(row) for row in annotations}
    dev_ids = {canonical_id(row) for row in dev}
    if train_ids != annotation_ids:
        raise ValueError("Training annotation/transcript ID sets differ")
    if train_ids & dev_ids:
        raise ValueError("Canonical train/dev video IDs overlap")
    if {transcript_digest(row) for row in texts} & {transcript_digest(row) for row in dev}:
        raise ValueError("Normalized train/dev transcripts overlap")
    manifest = {
        "training_videos": len(train_ids), "development_videos": len(dev_ids),
        "dev_used_for_gradient_training": False,
        "canonical_id_overlap": 0, "normalized_transcript_overlap": 0,
        "source_sha256": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for path in (train_text, train_annotations, dev_text)},
        "limitation": "Exact normalized transcript checks do not exclude all near duplicates or establish historical checkpoint provenance.",
    }
    return texts, annotations, manifest

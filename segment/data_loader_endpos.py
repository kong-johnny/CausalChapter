# -*- coding: utf-8 -*-
"""
Data loading for boundary prediction using utterance END-position labeling.

Each utterance is encoded as BOS + utterance tokens, but the classification
position is the last token of the utterance so a causal LM can attend to the
current utterance text before deciding whether it ends a chapter.
"""
import os
import json
import torch
from torch.utils.data import Dataset


LABEL_MAP = {"B-EOP": 1, "O": 0, 1: 1, 0: 0, "1": 1, "0": 0}


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_subheading_and_labels(jsonl_path):
    """Load subheading_abstract JSONL and extract boundary labels.

    Each entry has 'subheading_abstract' list. The last utterance in each
    segment is a boundary (B-EOP), others are non-boundary (O).

    Returns: {example_id: [0, 0, 1, 0, 0, 1, ...]}  (1 = B-EOP boundary)
    """
    gt_labels = {}
    data = load_jsonl(jsonl_path)
    for item in data:
        eid = item["example_id"]
        subheadings = item.get("subheading_abstract", [])

        # Count utterances per segment by matching with the text file
        # We need the text file separately to know total utterance count
        # For now, store the segments for later matching
        gt_labels[eid] = {
            "segments": subheadings,
        }
    return gt_labels


def _time_to_seconds(t):
    """Convert time string (HH:MM:SS or float) to seconds."""
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, str):
        parts = t.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        return float(t)
    return 0.0


def build_boundary_labels(text_data, gt_data):
    """Build per-utterance boundary labels from GT segment end times.

    IMPORTANT: in the AVLecture ``subheading_abstract`` files, ``index`` is the
    chapter/segment ordinal (1, 2, 3, ...), *not* an utterance index.  The old
    implementation trusted ``index`` first, which created bogus labels at the
    first few utterances of every video and inflated segmentation F1 while
    producing unusable predicted chapters.

    We therefore prefer ``seg['end']`` and align it to the utterance whose end
    timestamp is closest.  ``index`` is used only as a last-resort fallback when
    timestamps/stet are unavailable.  The final video end is kept as B-EOP; it
    is harmless for segment construction because the generator already appends
    the video end, and it keeps training/eval semantics consistent.
    """
    labels = {}
    for eid, td in text_data.items():
        if eid not in gt_data:
            continue
        n_utts = len(td["text"])
        lbl = [0] * n_utts
        stet = td.get("stet", [])

        for seg in gt_data[eid]["segments"]:
            best_idx = -1

            # Preferred path: map GT segment end time to nearest utterance end.
            if stet and seg.get("end") is not None:
                seg_end = _time_to_seconds(seg.get("end", "0"))
                best_dist = float("inf")
                for i, se in enumerate(stet):
                    if not isinstance(se, (list, tuple)) or len(se) < 2:
                        continue
                    u_e = _time_to_seconds(se[1])
                    dist = abs(u_e - seg_end)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i

            # Fallback only: some datasets may really store 1-based utterance end
            # indices; AVLecture does not, but this keeps the loader generic.
            if best_idx < 0:
                index = seg.get("index")
                if index is not None:
                    idx = int(index) - 1
                    if 0 <= idx < n_utts:
                        best_idx = idx

            if 0 <= best_idx < n_utts:
                lbl[best_idx] = 1

        labels[eid] = lbl
    return labels


class BoundaryDataset(Dataset):
    """Dataset for boundary prediction with Qwen2.5-3B.

    Each sample is a window of utterances. BOS token is added per utterance.
    Labels are applied at the END token position of each utterance.
    Field names keep bos_positions/n_valid_bos for compatibility with existing model/training code.
    """

    def __init__(
        self,
        text_jsonl,
        gt_jsonl,
        tokenizer,
        max_seq_length=4096,
        split="train",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split

        # Load text data (utterances)
        text_data = {}
        for item in load_jsonl(text_jsonl):
            text_data[item["example_id"]] = item

        # Load GT labels
        gt_raw = load_subheading_and_labels(gt_jsonl)
        self.labels = build_boundary_labels(text_data, gt_raw)

        # Build samples: each is a contiguous window of utterances
        self.samples = []
        self._build_samples(text_data)

        print(f"[DATASET] {split}: {len(self.samples)} windows "
              f"from {len({s['eid'] for s in self.samples})} videos")

    def _build_samples(self, text_data):
        """Build overlapping window samples from video data."""
        for eid, lbl in self.labels.items():
            if eid not in text_data:
                continue

            td = text_data[eid]
            utts = td["text"]
            n = len(utts)
            if n == 0:
                continue

            # Process each video with sliding windows
            sent_idx = 0
            while sent_idx < n:
                # Try to fit as many utterances as possible in one window
                tokens_so_far = 0
                window_end = sent_idx
                for i in range(sent_idx, n):
                    # Each utterance: BOS + utterance tokens
                    utt_text = utts[i]
                    utt_tokens = self.tokenizer(
                        self.tokenizer.bos_token + utt_text,
                        add_special_tokens=False,
                    )["input_ids"]
                    if tokens_so_far + len(utt_tokens) <= self.max_seq_length - 1:
                        tokens_so_far += len(utt_tokens)
                        window_end = i + 1
                    else:
                        break

                # Need at least one utterance
                if window_end <= sent_idx:
                    window_end = min(sent_idx + 1, n)

                # Get labels for this window
                window_labels = lbl[sent_idx:window_end]

                # Overlap: share one sentence with next window
                self.samples.append({
                    "eid": eid,
                    "utt_start": sent_idx,
                    "utt_end": window_end,
                    "utterances": utts[sent_idx:window_end],
                    "labels": window_labels,
                    "n_utts": n,
                })

                if window_end >= n:
                    break

                # Sliding: share last sentence
                sent_idx = max(sent_idx + 1, window_end - 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        utts = sample["utterances"]
        lbl = sample["labels"]

        # Build input: prepend BOS token to each utterance, classify at utterance END.
        # For a causal LM, hidden state at BOS cannot see the current utterance; the
        # last token can attend to BOS+utterance+previous context.  Keep the legacy
        # variable/key name `bos_positions` to avoid touching model/train code.
        all_input_ids = []
        bos_positions = []  # actually end-token positions
        kept_label_indices = []

        for i, u in enumerate(utts):
            tok_ids = self.tokenizer(
                self.tokenizer.bos_token + u,
                add_special_tokens=False,
            )["input_ids"]
            if not tok_ids:
                tok_ids = [self.tokenizer.bos_token_id or self.tokenizer.eos_token_id]
            # Window builder should already fit, but guard truncation precisely.
            if len(all_input_ids) + len(tok_ids) > self.max_seq_length:
                remain = self.max_seq_length - len(all_input_ids)
                if remain <= 0:
                    break
                tok_ids = tok_ids[:remain]
            all_input_ids.extend(tok_ids)
            bos_positions.append(len(all_input_ids) - 1)
            kept_label_indices.append(i)
            if len(all_input_ids) >= self.max_seq_length:
                break

        # Pad to max_seq_length
        attention_mask = [1] * len(all_input_ids)
        while len(all_input_ids) < self.max_seq_length:
            all_input_ids.append(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            attention_mask.append(0)

        # Pad positions and labels
        n_bos = len(bos_positions)
        labels = [lbl[i] for i in kept_label_indices]
        while len(labels) < self.max_seq_length:
            labels.append(-100)
            bos_positions.append(0)  # dummy position

        max_bos = self.max_seq_length
        bos_positions = bos_positions[:max_bos]
        labels = labels[:max_bos]

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "bos_positions": torch.tensor(bos_positions, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eid": sample["eid"],
            "utt_start": sample["utt_start"],
            "utt_end": sample["utt_end"],
            "n_utts": sample["n_utts"],
            "n_valid_bos": n_bos,
        }
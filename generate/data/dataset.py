# -*- coding: utf-8 -*-
"""JointDataset"""
from collections import defaultdict
import torch
from torch.utils.data import Dataset

from ..config.fewshot import FEWSHOT_EXAMPLES
from ..data.prompt_builder import build_prompt_parts
from ..data.vis_features import load_vis_seq
from ..utils.text_utils import build_prompt_ids_with_truncation


class JointDataset(Dataset):
    def __init__(
        self,
        samples,
        tokenizer,
        vis_index,
        vis_dim=512,
        max_length=4096,
        seg_len=8000,
        window_size=7,
        vis_token_count=4,
        prompt_mode="pass2",
        use_fewshot=True,
        fewshot_num=2,
        debug=False,
    ):
        self.samples = samples
        self.prompt_mode = prompt_mode
        self.use_fewshot = use_fewshot
        self.fewshot_num = fewshot_num
        self.tokenizer = tokenizer
        self.vis_index = vis_index
        self.vis_dim = vis_dim
        self.max_length = max_length
        self.seg_len = seg_len
        self.window_size = window_size
        self.vis_token_count = vis_token_count
        self.fewshot_examples = FEWSHOT_EXAMPLES[:fewshot_num] if use_fewshot else None
        self.debug = debug

        self.by_eid = defaultdict(list)
        for i, s in enumerate(self.samples):
            self.by_eid[s["example_id"]].append((i, s))
        for eid in self.by_eid:
            self.by_eid[eid].sort(key=lambda x: (x[1]["start_idx"], x[1]["end_idx"]))

        self.rank_map = {}
        for eid, arr in self.by_eid.items():
            for rank, (idx, _) in enumerate(arr):
                self.rank_map[idx] = rank

    def __len__(self):
        return len(self.samples)

    def _get_prev_titles(self, idx):
        s = self.samples[idx]
        eid = s["example_id"]
        arr = self.by_eid[eid]
        cur_rank = self.rank_map[idx]
        out = []
        for r in range(max(0, cur_rank - self.window_size), cur_rank):
            _, prev_s = arr[r]
            out.append(prev_s["target_subheading"])
        return out

    def __getitem__(self, idx):
        s = self.samples[idx]

        # ---- Sandwich truncation: split into head / middle / tail ----
        # head: system + VIS + few-shot + context (NEVER truncated)
        # middle: Segment:\n{transcript}\n (left-truncated if needed)
        # tail: <|im_end|>\n<|im_start|>assistant\n (NEVER truncated)
        head, middle, tail = build_prompt_parts(
            segment_text=s["segment_text"],
            context_subheadings=s.get("causal_context_titles", []),
            fewshot_examples=self.fewshot_examples if self.use_fewshot else None,
            num_query_tokens=self.vis_token_count,
            pass1_self_title=(s.get("pass1_title") if self.prompt_mode == "pass2" else None),
            tokenizer=self.tokenizer,
        )

        # ---- Tokenize target independently (cannot be truncated) ----
        target_text = s["target_subheading"] + "<|im_end|>"
        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]

        # ---- Reserve space for target; compute effective budget for prompt ----
        SAFETY_BUF = 16
        effective_max = self.max_length - len(target_ids) - SAFETY_BUF

        if effective_max <= 0:
            raise ValueError(
                f"max_length({self.max_length}) - target_ids({len(target_ids)}) - "
                f"safety_buf({SAFETY_BUF}) <= 0. Increase max_length or reduce target size."
            )

        # ---- Build truncated prompt IDs using sandwich truncation ----
        # build_prompt_ids_with_truncation returns (final_ids, head_ids, middle_ids, tail_ids, budget_used)
        prompt_and_ids = build_prompt_ids_with_truncation(
            self.tokenizer, head, middle, tail,
            max_total_tokens=effective_max,
            safety_margin=0,  # budget already accounts for everything
        )
        prompt_ids, head_ids, middle_ids, tail_ids, budget_used = prompt_and_ids

        # ---- Sanity: if head+tail+target itself exceeds max_length, fail fast ----
        total_needed = len(prompt_ids) + len(target_ids)
        if total_needed > self.max_length:
            raise ValueError(
                f"head({len(head_ids)}) + middle_truncated({len(middle_ids)}) + "
                f"tail({len(tail_ids)}) + target({len(target_ids)}) = {total_needed} "
                f"exceeds max_length({self.max_length}). "
                f"Reduce VIS/few-shot/context or increase max_length."
            )

        # ---- Assemble final sequences ----
        input_ids = prompt_ids + target_ids
        # Labels: -100 for prompt part, actual token IDs for target part
        labels = [-100] * len(prompt_ids) + target_ids

        # ---- Truncate to max_length only as a safety net (shouldn't happen) ----
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        # ---- Padding ----
        pad_len = self.max_length - len(input_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        attention_mask = [1] * len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            labels += [-100] * pad_len
            attention_mask += [0] * pad_len

        # ---- Visual features ----
        vis_seq = load_vis_seq(
            self.vis_index,
            s["example_id"],
            s["start_idx"],
            s["end_idx"],
            vis_dim=self.vis_dim,
            start_time=s.get("start_time"),
            end_time=s.get("end_time"),
        )
        if vis_seq is None:
            vis_seq = torch.zeros(1, self.vis_dim, dtype=torch.float)
        elif not isinstance(vis_seq, torch.Tensor):
            vis_seq = torch.tensor(vis_seq, dtype=torch.float)

        # ---- Debug output for first 3 samples ----
        if self.debug and idx < 3:
            print(f"\n{'='*80}")
            print(f"[DEBUG dataset[{idx}]]")
            print(f"  head tokens : {len(head_ids)}")
            print(f"  middle tokens: {len(middle_ids)} (budget used: {budget_used})")
            print(f"  tail tokens : {len(tail_ids)}")
            print(f"  prompt_ids  : {len(prompt_ids)}")
            print(f"  target_ids  : {len(target_ids)}")
            print(f"  input_ids   : {len(input_ids)} (max_length={self.max_length})")
            print(f"  labels -100 count: {sum(1 for x in labels if x == -100)}")
            print(f"  target_text : {repr(target_text[:80])}")
            print(f"  tail decoded: {repr(self.tokenizer.decode(tail_ids))}")
            # Show end of decoded prompt
            full_prompt_decoded = self.tokenizer.decode(prompt_ids)
            print(f"  prompt tail (last 150 chars): {repr(full_prompt_decoded[-150:])}")
            print(f"{'='*80}\n")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "vis_seq": vis_seq,
        }

# -*- coding: utf-8 -*-
"""JointCollator"""
import numpy as np
import torch


class JointCollator:
    def __init__(self, tokenizer, vis_dim=512):
        self.tokenizer = tokenizer
        self.vis_dim = vis_dim

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        max_t = max(x["vis_seq"].shape[0] for x in batch)

        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []
        vis_seqs = []
        vis_masks = []
        metas = []

        for x in batch:
            l = len(x["input_ids"])
            pad_l = max_len - l

            input_ids.append(x["input_ids"].tolist() + [pad_id] * pad_l)
            attention_mask.append(x["attention_mask"].tolist() + [0] * pad_l)
            labels.append(x["labels"].tolist() + [-100] * pad_l)

            vis_np = x["vis_seq"].detach().cpu().numpy() if isinstance(x["vis_seq"], torch.Tensor) else np.asarray(x["vis_seq"], dtype=np.float32)
            t = vis_np.shape[0]
            pad_t = max_t - t
            if pad_t > 0:
                pad_arr = np.zeros((pad_t, self.vis_dim), dtype=np.float32)
                vis = np.concatenate([vis_np, pad_arr], axis=0)
            else:
                vis = vis_np
            vis_seqs.append(vis.astype(np.float32, copy=False))
            vis_masks.append([1] * t + [0] * pad_t)

            # metas.append(x["meta"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "vis_seq": torch.tensor(np.stack(vis_seqs, axis=0), dtype=torch.float32),
            "vis_mask": torch.tensor(vis_masks, dtype=torch.long),
            # "meta": metas,
        }
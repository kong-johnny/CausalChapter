# -*- coding: utf-8 -*-
"""Paper-aligned CSSE: compact candidates plus removal-intervention KL."""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..data.prompt_builder import build_prompt_parts
from ..inference.generate import _pad_visual_batch, generate_one
from ..utils.text_utils import build_prompt_with_truncation, postprocess_title


class KLSupportRetriever:
    """Select historical contexts by prediction-level removal influence.

    MiniLM is used only for compact candidate construction. The final support
    score is exclusively the aligned token-distribution KL divergence.
    """

    def __init__(
        self,
        generation_model,
        tokenizer,
        device="cuda",
        near_size=3,
        semantic_size=10,
        final_topk=5,
        intervention_tokens=16,
        variant_microbatch=2,
        sentence_encoder_path=None,
        vis_dim=512,
        vis_token_count=4,
        input_max_tokens=8192,
    ):
        if not hasattr(generation_model, "forward_logits_at_positions"):
            raise TypeError(
                "KLSupportRetriever requires FastMultiModalCausalLM, not the "
                "bare `.llm` module."
            )
        self.model = generation_model
        self.tokenizer = tokenizer
        self.device = device
        self.near_size = near_size
        self.semantic_size = semantic_size
        self.final_topk = final_topk
        self.intervention_tokens = intervention_tokens
        self.variant_microbatch = max(1, variant_microbatch)
        self.vis_dim = vis_dim
        self.vis_token_count = vis_token_count
        self.input_max_tokens = input_max_tokens
        self.sentence_encoder_path = sentence_encoder_path or os.environ.get(
            "SENTENCE_ENCODER_PATH",
            "/mnt/nas2/duanmo/weights/all-MiniLM-L6-v2",
        )
        self._encoder = None
        self._video_key = None
        self._segments = None
        self._embeddings = None
        self.stats = {
            "reference_generations": 0,
            "teacher_forced_forwards": 0,
            "candidate_variants": 0,
        }

    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0

    def _get_encoder(self):
        if self._encoder is None:
            # Prefer the original MiniLM/SentenceTransformer encoder when the
            # local weight directory is available.  Some training environments
            # do not mount /mnt/nas2/duanmo/weights/all-MiniLM-L6-v2; in that
            # case we fall back to a deterministic local hashing-TFIDF encoder
            # so KL-CSSE pass2 can still build causal support candidates instead
            # of crashing after pass1.
            if self.sentence_encoder_path and os.path.exists(self.sentence_encoder_path):
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(
                    self.sentence_encoder_path,
                    device="cpu",
                )
            else:
                print(
                    f"[WARN] sentence encoder path not found: {self.sentence_encoder_path}; "
                    "using local HashingVectorizer fallback for semantic candidates."
                )

                class _HashingEncoder:
                    def __init__(self, n_features=512):
                        from sklearn.feature_extraction.text import HashingVectorizer

                        self.vectorizer = HashingVectorizer(
                            n_features=n_features,
                            alternate_sign=False,
                            norm="l2",
                            ngram_range=(1, 2),
                            lowercase=True,
                        )

                    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
                        x = self.vectorizer.transform(texts).astype(np.float32)
                        arr = x.toarray().astype(np.float32)
                        if normalize_embeddings:
                            norms = np.linalg.norm(arr, axis=1, keepdims=True)
                            arr = arr / np.maximum(norms, 1e-12)
                        return arr

                self._encoder = _HashingEncoder()
        return self._encoder

    @staticmethod
    def _representation_text(segment):
        span = segment.get("time_span") or []
        span_text = " ".join(str(v) for v in span)
        return " ".join(
            part
            for part in [
                segment.get("text", ""),
                segment.get("title", ""),
                span_text,
            ]
            if part
        )

    def ensure_built(self, all_segments, video_key=None):
        if video_key == self._video_key and self._embeddings is not None:
            return
        texts = [self._representation_text(seg) for seg in all_segments]
        embeddings = self._get_encoder().encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self._video_key = video_key
        self._segments = all_segments
        self._embeddings = np.asarray(embeddings, dtype=np.float32)

    def candidate_indices(self, current_idx, candidate_size=None):
        if current_idx <= 0:
            return []
        historical = list(range(current_idx))
        near = historical[-self.near_size :] if self.near_size > 0 else []
        semantic = []
        if self._embeddings is not None and self.semantic_size > 0:
            scores = self._embeddings[:current_idx] @ self._embeddings[current_idx]
            order = np.argsort(-scores)
            semantic = [int(i) for i in order[: self.semantic_size]]

        # Keep nearby contexts and semantic contexts, then restore chronology.
        merged = sorted(set(near + semantic))
        if candidate_size is not None and len(merged) > candidate_size:
            near_set = set(near)
            semantic_rank = {idx: rank for rank, idx in enumerate(semantic)}
            prioritized = sorted(
                merged,
                key=lambda idx: (
                    0 if idx in near_set else 1,
                    -idx if idx in near_set else semantic_rank.get(idx, 10**9),
                ),
            )[:candidate_size]
            merged = sorted(prioritized)
        return merged

    def _prompt(
        self,
        segment_text,
        pass1_title,
        context_titles,
        fewshot_examples,
    ):
        head, middle, tail = build_prompt_parts(
            segment_text=segment_text,
            context_subheadings=context_titles,
            fewshot_examples=fewshot_examples,
            max_context=None,
            num_query_tokens=self.vis_token_count,
            pass1_self_title=pass1_title,
            tokenizer=self.tokenizer,
        )
        return build_prompt_with_truncation(
            self.tokenizer,
            head,
            middle,
            tail,
            max_total_tokens=self.input_max_tokens,
            safety_margin=max(32, self.intervention_tokens + 8),
        )

    def _target_logits(self, prompts, target_ids, vis_seq_np):
        """Compute target-position logits with right-padded teacher forcing."""
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        prompt_ids = [
            self.tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts
        ]
        sequences = [ids + target_ids for ids in prompt_ids]
        max_len = max(len(ids) for ids in sequences)
        input_ids = []
        masks = []
        positions = []
        target_len = len(target_ids)
        for pids, sequence in zip(prompt_ids, sequences):
            padding = max_len - len(sequence)
            input_ids.append(sequence + [pad_id] * padding)
            masks.append([1] * len(sequence) + [0] * padding)
            first = len(pids) - 1
            positions.append(list(range(first, first + target_len)))
        ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(masks, dtype=torch.long, device=self.device)
        pos_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)
        vis_batch, vis_mask = _pad_visual_batch(
            [vis_seq_np] * len(prompts),
            self.vis_dim,
            self.device,
            getattr(self.model, "disable_visual", False),
        )
        with torch.inference_mode():
            logits = self.model.forward_logits_at_positions(
                input_ids=ids_tensor,
                attention_mask=mask_tensor,
                logit_positions=pos_tensor,
                vis_seq=vis_batch,
                vis_mask=vis_mask,
            )
        self.stats["teacher_forced_forwards"] += 1
        return logits.float()

    def select_contexts(
        self,
        current_segment: Dict,
        all_segments: Sequence[Dict],
        current_idx: int,
        vis_seq_np,
        *,
        candidate_size=10,
        topk=None,
        fewshot_examples=None,
        gen_kwargs=None,
    ) -> Tuple[List[int], List[Tuple[int, float, Dict]]]:
        candidate_ids = self.candidate_indices(current_idx, candidate_size)
        if not candidate_ids:
            return [], []
        topk = self.final_topk if topk is None else topk
        titles = {
            idx: postprocess_title(all_segments[idx].get("title", ""))
            for idx in candidate_ids
        }
        full_titles = [titles[idx] for idx in candidate_ids if titles[idx]]
        full_prompt = self._prompt(
            current_segment.get("text", ""),
            current_segment.get("title", ""),
            full_titles,
            fewshot_examples,
        )

        # This short full-context output supplies one shared prefix for aligned
        # full/removal distributions. Leave-one-out variants are not decoded.
        # gen_kwargs may carry final-decoding max_new_tokens from eval_env; do not
        # let it collide with / override the retrieval intervention length here.
        ref_gen_kwargs = dict(gen_kwargs or {})
        ref_gen_kwargs.pop("max_new_tokens", None)
        reference = generate_one(
            self.model,
            self.tokenizer,
            full_prompt,
            vis_seq_np,
            self.device,
            vis_dim=self.vis_dim,
            max_new_tokens=self.intervention_tokens,
            **ref_gen_kwargs,
        )
        self.stats["reference_generations"] += 1
        reference = reference or current_segment.get("title", "")
        target_ids = self.tokenizer(
            reference,
            add_special_tokens=False,
        )["input_ids"][: self.intervention_tokens]
        if not target_ids:
            return candidate_ids[-topk:], [
                (idx, 0.0, {"kl": 0.0}) for idx in candidate_ids
            ]

        full_logits = self._target_logits([full_prompt], target_ids, vis_seq_np)[0]
        full_logp = F.log_softmax(full_logits, dim=-1)
        full_prob = full_logp.exp()
        details = []
        for start in range(0, len(candidate_ids), self.variant_microbatch):
            chunk = candidate_ids[start : start + self.variant_microbatch]
            prompts = []
            for removed in chunk:
                remaining = [
                    titles[idx]
                    for idx in candidate_ids
                    if idx != removed and titles[idx]
                ]
                prompts.append(
                    self._prompt(
                        current_segment.get("text", ""),
                        current_segment.get("title", ""),
                        remaining,
                        fewshot_examples,
                    )
                )
            removal_logits = self._target_logits(prompts, target_ids, vis_seq_np)
            removal_logp = F.log_softmax(removal_logits, dim=-1)
            kl = (full_prob.unsqueeze(0) * (
                full_logp.unsqueeze(0) - removal_logp
            )).sum(dim=-1).mean(dim=-1)
            for idx, value in zip(chunk, kl.detach().cpu().tolist()):
                details.append((idx, float(value), {"kl": float(value)}))
            del removal_logits, removal_logp, kl

        self.stats["candidate_variants"] += len(candidate_ids)
        details.sort(key=lambda row: row[1], reverse=True)
        selected = sorted(idx for idx, _, _ in details[:topk])
        return selected, details

    # Reject accidental use of the legacy retrieval API: it cannot receive the
    # current visual input and therefore cannot compute paper-aligned KL.
    def retrieve(self, *args, **kwargs):
        return []  # legacy debug only; KL inference uses select_contexts()
        raise RuntimeError(
            "Use select_contexts(); the legacy retrieve() API implements the "
            "old light/Granger path and lacks visual inputs for KL-CSSE."
        )

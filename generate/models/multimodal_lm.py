# -*- coding: utf-8 -*-
"""MultiModalCausalLM with pluggable fusion modules.

Fusion modes (set via model_config or init args):
  'cross_attn'  — default: QFormer -> inject vis tokens at <VIS_x> positions (baseline)
  'linear'      — LinearFusion: concat or gated-add vis+text then project
  'moe'         — MoEFusion: Mixture-of-Experts (text-only / vis-only / cross)
  'text_only'   — ASR-Only: disable_visual=True, no QFormer
  'vis_only'    — Visual-Only: zero-out text embeddings, only use vis tokens + LLM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion_modules import LinearFusion, MoEFusion


class MultiModalCausalLM(nn.Module):
    def __init__(
        self,
        llm,
        qformer,
        vis_token_ids,
        label_smoothing=0.0,
        use_quantization=False,
        fusion_mode="cross_attn",
        llm_hidden=None,
        vis_hidden=512,
        num_query_tokens=4,
        linear_fusion_type="add",
        moe_num_experts=3,
        moe_top_k=2,
    ):
        super().__init__()
        self.llm = llm
        self.qformer = qformer
        self.vis_token_ids = vis_token_ids
        self._warned_no_vis = False
        self.disable_visual = False
        self.label_smoothing = label_smoothing
        self.use_quantization = use_quantization
        self.fusion_mode = fusion_mode

        if llm_hidden is None:
            llm_hidden = llm.get_input_embeddings().weight.shape[1]
        self.llm_hidden = llm_hidden
        self.vis_hidden = vis_hidden
        self.num_query_tokens = num_query_tokens

        if fusion_mode == "linear":
            self.linear_fusion = LinearFusion(
                llm_hidden=llm_hidden,
                vis_hidden=vis_hidden,
                num_query_tokens=num_query_tokens,
                proj_type=linear_fusion_type,
            )
        elif fusion_mode == "moe":
            self.moe_fusion = MoEFusion(
                llm_hidden=llm_hidden,
                vis_hidden=vis_hidden,
                num_query_tokens=num_query_tokens,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
            )
        elif fusion_mode == "vis_only":
            self.vis_to_llm = nn.Sequential(
                nn.Linear(vis_hidden, llm_hidden),
                nn.GELU(),
                nn.Linear(llm_hidden, llm_hidden),
            )

    def to(self, *args, **kwargs):
        if self.use_quantization:
            self.qformer = self.qformer.to(*args, **kwargs)
            if hasattr(self, 'linear_fusion'):
                self.linear_fusion = self.linear_fusion.to(*args, **kwargs)
            if hasattr(self, 'moe_fusion'):
                self.moe_fusion = self.moe_fusion.to(*args, **kwargs)
            if hasattr(self, 'vis_to_llm'):
                self.vis_to_llm = self.vis_to_llm.to(*args, **kwargs)
            return self
        return super().to(*args, **kwargs)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def _inject_visual_embeds(self, input_ids, inputs_embeds, vis_seq, vis_mask):
        # Guard: ensure vis_seq has batch dim [B, T, V], vis_mask is tensor
        if vis_seq.dim() == 1:
            vis_seq = vis_seq.unsqueeze(0)  # [V] -> [1, V]
        elif vis_seq.dim() == 2:
            vis_seq = vis_seq.unsqueeze(0)  # [T, V] -> [1, T, V]
        if vis_mask is None or (isinstance(vis_mask, torch.Tensor) and vis_mask.numel() == 0):
            vis_mask = torch.ones(vis_seq.shape[0], vis_seq.shape[1], device=vis_seq.device)
        if isinstance(vis_mask, bool):
            vis_mask = torch.tensor([[vis_mask]], device=vis_seq.device)
        if vis_mask.dim() == 1:
            vis_mask = vis_mask.unsqueeze(0)

        vis_tokens = self.qformer(vis_seq, vis_mask)

        if not hasattr(self, '_printed_vis_stats'):
            text_emb = inputs_embeds
            text_norm = text_emb.norm(dim=-1).mean().item()
            text_std = text_emb.std().item()
            text_mean = text_emb.mean().item()
            vis_norm = vis_tokens.norm(dim=-1).mean().item()
            vis_std = vis_tokens.std().item()
            vis_mean = vis_tokens.mean().item()
            print(f"[VIS STATS] text_emb: mean={text_mean:.4f}, std={text_std:.4f}, norm={text_norm:.4f}")
            print(f"[VIS STATS] vis_tokens: mean={vis_mean:.4f}, std={vis_std:.4f}, norm={vis_norm:.4f}")
            print(f"[VIS STATS] ratio (vis/text norm): {vis_norm / max(text_norm, 1e-6):.2f}x")
            self._printed_vis_stats = True

        B, Q, _ = vis_tokens.shape
        vis_tokens = vis_tokens.to(dtype=inputs_embeds.dtype)
        replaced = 0
        for qi, token_id in enumerate(self.vis_token_ids):
            if qi >= Q:
                break
            mask = (input_ids == token_id)
            for b in range(B):
                pos = torch.where(mask[b])[0]
                if len(pos) > 0:
                    inputs_embeds[b, pos[0], :] = vis_tokens[b, qi, :]
                    replaced += 1
        if replaced == 0 and not self._warned_no_vis:
            print("[WARN] MultiModalCausalLM: no <VIS_x> tokens found in input_ids. "
                  "Visual features NOT injected. Check prompt_builder & tokenizer.")
            self._warned_no_vis = True
        return inputs_embeds, None

    def _forward_linear(self, input_ids, inputs_embeds, vis_seq, vis_mask):
        vis_seq = vis_seq.to(dtype=inputs_embeds.dtype)
        # Cast fusion module to input dtype (LLM may use bf16 while fusion is fp32)
        self.linear_fusion = self.linear_fusion.to(dtype=inputs_embeds.dtype)
        fused_emb = self.linear_fusion(inputs_embeds, vis_seq, vis_mask)  # [B, L, H]
        B, L, H = inputs_embeds.shape
        # Replace VIS_0 position with fused embedding
        vis0_mask = (input_ids == self.vis_token_ids[0]).unsqueeze(-1)  # [B, L, 1]
        # Zero out remaining VIS_1..VIS_N positions
        other_vis_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for qi in range(1, len(self.vis_token_ids)):
            other_vis_mask |= (input_ids == self.vis_token_ids[qi])
        out = torch.where(vis0_mask, fused_emb.to(dtype=inputs_embeds.dtype), inputs_embeds)
        out = torch.where(other_vis_mask.unsqueeze(-1), 0.0, out)
        return out

    def _forward_moe(self, input_ids, inputs_embeds, vis_seq, vis_mask):
        vis_seq = vis_seq.to(dtype=inputs_embeds.dtype)
        # Cast fusion module to input dtype (LLM may use bf16 while fusion is fp32)
        self.moe_fusion = self.moe_fusion.to(dtype=inputs_embeds.dtype)
        fused_emb = self.moe_fusion(inputs_embeds, vis_seq, vis_mask)  # [B, L, H]
        B, L, H = inputs_embeds.shape
        vis0_mask = (input_ids == self.vis_token_ids[0]).unsqueeze(-1)
        other_vis_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for qi in range(1, len(self.vis_token_ids)):
            other_vis_mask |= (input_ids == self.vis_token_ids[qi])
        out = torch.where(vis0_mask, fused_emb, inputs_embeds)
        out = torch.where(other_vis_mask.unsqueeze(-1), 0.0, out)
        return out

    def _forward_vis_only(self, input_ids, inputs_embeds, vis_seq, vis_mask):
        B, L, H = inputs_embeds.shape
        vis_seq = vis_seq.to(dtype=inputs_embeds.dtype)
        self.vis_to_llm = self.vis_to_llm.to(dtype=inputs_embeds.dtype)
        if vis_mask is not None:
            mask = vis_mask.unsqueeze(-1).to(dtype=inputs_embeds.dtype)
            vis_pooled = (vis_seq * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            vis_pooled = vis_seq.mean(dim=1)
        vis_proj = self.vis_to_llm(vis_pooled)
        inputs_embeds = vis_proj.unsqueeze(1).expand(B, L, -1)
        return inputs_embeds

    def forward(self, input_ids, attention_mask, labels=None, vis_seq=None, vis_mask=None):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids).clone()
        # Cast to LLM dtype (get_input_embeddings returns float32 but LLM may be bf16)
        llm_dtype = next(self.llm.parameters()).dtype
        inputs_embeds = inputs_embeds.to(dtype=llm_dtype)

        if vis_seq is not None and vis_mask is not None and not self.disable_visual:
            if self.fusion_mode == "text_only":
                for qi in self.vis_token_ids:
                    inputs_embeds[input_ids == qi] = 0.0
            elif self.fusion_mode == "vis_only":
                inputs_embeds = self._forward_vis_only(input_ids, inputs_embeds, vis_seq, vis_mask)
            elif self.fusion_mode == "linear":
                inputs_embeds = self._forward_linear(input_ids, inputs_embeds, vis_seq, vis_mask)
            elif self.fusion_mode == "moe":
                inputs_embeds = self._forward_moe(input_ids, inputs_embeds, vis_seq, vis_mask)
            else:
                inputs_embeds, _ = self._inject_visual_embeds(input_ids, inputs_embeds, vis_seq, vis_mask)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        if self.label_smoothing > 0 and labels is not None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100, label_smoothing=self.label_smoothing,
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
            )
            return type('obj', (object,), {'loss': loss, 'logits': outputs.logits})()
        return outputs

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, vis_seq=None, vis_mask=None, **kwargs):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids).clone()
        llm_dtype = next(self.llm.parameters()).dtype
        inputs_embeds = inputs_embeds.to(dtype=llm_dtype)
        if vis_seq is not None and vis_mask is not None and not self.disable_visual:
            if self.fusion_mode == "text_only":
                for qi in self.vis_token_ids:
                    inputs_embeds[input_ids == qi] = 0.0
            elif self.fusion_mode == "vis_only":
                inputs_embeds = self._forward_vis_only(input_ids, inputs_embeds, vis_seq, vis_mask)
            elif self.fusion_mode == "linear":
                inputs_embeds = self._forward_linear(input_ids, inputs_embeds, vis_seq, vis_mask)
            elif self.fusion_mode == "moe":
                inputs_embeds = self._forward_moe(input_ids, inputs_embeds, vis_seq, vis_mask)
            else:
                inputs_embeds, _ = self._inject_visual_embeds(input_ids, inputs_embeds, vis_seq, vis_mask)

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
# -*- coding: utf-8 -*-
"""Visual-Text Fusion Modules for CausalChapter ablation study.

Supports 5 experiment groups:
1. ASR-Only (text-only, no visual features)
2. Visual-Only (visual features only, no text context)
3. ASR+Visual Cross-Attention (baseline, existing)
4. ASR+Visual Linear Fusion (new)
5. ASR+Visual Mixture-of-Experts (new)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearFusion(nn.Module):
    """Linear fusion: concatenate text and visual embeddings, project to LLM space.

    Args:
        llm_hidden: hidden size of the LLM
        vis_hidden: hidden size of visual features (default 512)
        num_query_tokens: number of visual query tokens (default 4)
        proj_type: 'concat' or 'add'
    """
    def __init__(self, llm_hidden, vis_hidden=512, num_query_tokens=4, proj_type="concat"):
        super().__init__()
        self.llm_hidden = llm_hidden
        self.num_query_tokens = num_query_tokens
        self.proj_type = proj_type

        if proj_type == "concat":
            # Concatenate text embedding + vis embedding for each token position
            # then project back to llm_hidden
            self.proj = nn.Sequential(
                nn.Linear(llm_hidden + vis_hidden, llm_hidden),
                nn.GELU(),
                nn.Linear(llm_hidden, llm_hidden),
            )
        elif proj_type == "add":
            # Project visual features to text space, then add
            self.vis_proj = nn.Linear(vis_hidden, llm_hidden)
            self.gate = nn.Linear(llm_hidden, 1)  # per-token gating
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

    def forward(self, text_emb, vis_tokens, vis_mask=None):
        """
        Args:
            text_emb: [B, L, H_llm]
            vis_tokens: [B, Q, H_vis]
            vis_mask: [B, Q] (1=valid, 0=padding)
        Returns:
            fused_emb: [B, L, H_llm]
        """
        B, L, H = text_emb.shape
        Q = vis_tokens.shape[1]

        if self.proj_type == "concat":
            # For each text position, we need Q visual tokens
            # Expand: text [B, L, 1, H] + vis [B, 1, Q, H_vis]
            text_expanded = text_emb.unsqueeze(2).expand(B, L, Q, H)
            vis_expanded = vis_tokens.unsqueeze(1).expand(B, L, Q, -1)

            # Concatenate and project
            concat = torch.cat([text_expanded, vis_expanded], dim=-1)  # [B, L, Q, H+H_vis]
            fused = self.proj(concat)  # [B, L, Q, H]

            # Aggregate over Q visual tokens (mean pooling)
            if vis_mask is not None:
                mask = vis_mask.unsqueeze(1).unsqueeze(-1).to(dtype=vis_tokens.dtype)  # [B, 1, Q, 1]
                fused = (fused * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
            else:
                fused = fused.mean(dim=2)

            return fused  # [B, L, H]

        elif self.proj_type == "add":
            # Project visual to text space
            vis_proj = self.vis_proj(vis_tokens)  # [B, Q, H]
            if vis_mask is not None:
                mask = vis_mask.unsqueeze(-1).to(dtype=vis_tokens.dtype)
                vis_proj = (vis_proj * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                vis_proj = vis_proj.mean(dim=1)  # [B, H]

            vis_pool = vis_proj.unsqueeze(1).expand(B, L, -1)  # [B, L, H]

            # Gating: decide how much visual info to inject per position
            gate = torch.sigmoid(self.gate(text_emb))  # [B, L, 1]
            fused = text_emb + gate * vis_pool
            return fused  # [B, L, H]


class MoEFusion(nn.Module):
    """Mixture-of-Experts fusion with 3 experts.

    Expert 0: Text-only
    Expert 1: Visual-only
    Expert 2: Cross-attention (concat + project)

    Gating network selects per-token expert weights.

    Args:
        llm_hidden: hidden size of the LLM
        vis_hidden: hidden size of visual features
        num_query_tokens: number of visual query tokens
        num_experts: number of experts (default 3)
        top_k: top-k experts to use (default 2)
    """
    def __init__(self, llm_hidden, vis_hidden=512, num_query_tokens=4, num_experts=3, top_k=2):
        super().__init__()
        self.llm_hidden = llm_hidden
        self.num_query_tokens = num_query_tokens
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert 0: Text-only projection
        self.expert_text = nn.Sequential(
            nn.Linear(llm_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )

        # Expert 1: Visual-only projection (pool Q features into 1 vector)
        self.expert_vis = nn.Sequential(
            nn.Linear(vis_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )

        # Expert 2: Cross-attention style (concat + project)
        self.expert_cross = nn.Sequential(
            nn.Linear(llm_hidden + vis_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(llm_hidden, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, num_experts),
        )

    def forward(self, text_emb, vis_tokens, vis_mask=None):
        """
        Args:
            text_emb: [B, L, H_llm]
            vis_tokens: [B, Q, H_vis]
            vis_mask: [B, Q] (1=valid, 0=padding)
        Returns:
            fused_emb: [B, L, H_llm]
        """
        B, L, H = text_emb.shape
        Q = vis_tokens.shape[1]

        # Compute expert outputs
        # Expert 0: text only
        expert_0_out = self.expert_text(text_emb)  # [B, L, H]

        # Expert 1: visual only (pool over segment dim → [B, vis_hidden])
        if vis_mask is not None:
            mask = vis_mask.unsqueeze(-1).to(dtype=vis_tokens.dtype)
            vis_pooled = (vis_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            vis_pooled = vis_tokens.mean(dim=1)
        expert_1_out = self.expert_vis(vis_pooled)  # [B, H]
        expert_1_out = expert_1_out.unsqueeze(1).expand(B, L, -1)  # [B, L, H]

        # Expert 2: cross (text + visual per position)
        text_expanded = text_emb.unsqueeze(2).expand(B, L, Q, H)
        vis_expanded = vis_tokens.unsqueeze(1).expand(B, L, Q, -1)
        concat = torch.cat([text_expanded, vis_expanded], dim=-1)
        if vis_mask is not None:
            mask_2d = vis_mask.unsqueeze(1).unsqueeze(-1).to(dtype=vis_tokens.dtype)
            expert_2_out = (self.expert_cross(concat) * mask_2d).sum(dim=2) / mask_2d.sum(dim=2).clamp(min=1)
        else:
            expert_2_out = self.expert_cross(concat).mean(dim=2)

        # Gating (softmax over experts)
        gate_logits = self.gate(text_emb)  # [B, L, num_experts]

        if self.num_experts > self.top_k:
            # Top-k gating
            topk_logits, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            gate_weights = F.softmax(topk_logits, dim=-1)

            # Gather expert outputs
            expert_outputs = torch.stack([expert_0_out, expert_1_out, expert_2_out], dim=-1)  # [B, L, H, 3]

            # Weighted sum of top-k experts
            result = torch.zeros_like(text_emb)
            for i in range(self.top_k):
                expert_idx = topk_indices[:, :, i]                              # [B, L]
                weight = gate_weights[:, :, i]                                   # [B, L]
                idx_4d = expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, 1)  # [B, L, H, 1]
                selected = torch.gather(expert_outputs, -1, idx_4d).squeeze(-1)  # [B, L, H]
                result = result + selected * weight.unsqueeze(-1)
        else:
            gate_weights = F.softmax(gate_logits, dim=-1)                        # [B, L, 3]
            expert_outputs = torch.stack([expert_0_out, expert_1_out, expert_2_out], dim=-1)  # [B, L, H, 3]
            result = torch.einsum("ble,blhe->blh", gate_weights, expert_outputs)  # [B, L, H]

        return result  # [B, L, H]
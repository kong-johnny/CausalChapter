# -*- coding: utf-8 -*-
"""SimpleQFormer"""
import torch
import torch.nn as nn


class SimpleQFormer(nn.Module):
    def __init__(
        self,
        vis_dim=512,
        q_hidden=512,
        llm_hidden=2048,
        num_query_tokens=4,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.vis_proj = nn.Linear(vis_dim, q_hidden)
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, q_hidden) * 0.02)

        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=q_hidden, num_heads=num_heads,
                dropout=dropout, batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(q_hidden) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(q_hidden, q_hidden * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(q_hidden * 4, q_hidden),
            )
            for _ in range(num_layers)
        ])
        self.norm2 = nn.ModuleList([nn.LayerNorm(q_hidden) for _ in range(num_layers)])
        self.to_llm = nn.Linear(q_hidden, llm_hidden)
        # Qwen embedding std ≈ 0.02, LayerNorm would force std=1 (50x too large!)
        # 用可学习的 per-token scale 代替 LayerNorm，初始化为 Qwen embedding std
        self.vis_scale = nn.Parameter(torch.ones(num_query_tokens) * 0.02)

    def forward(self, vis_seq, vis_mask):
        x = self.vis_proj(vis_seq)
        B = x.size(0)
        q = self.query_tokens.expand(B, -1, -1)

        key_padding_mask = ~(vis_mask.bool())

        for cross_attn, norm1_ffn, norm2_ffn in zip(
            self.cross_attn_layers, zip(self.norm1, self.ffn), zip(self.norm2, self.ffn)
        ):
            norm1, ffn = norm1_ffn
            norm2, ffn2 = norm2_ffn
            attn_out, _ = cross_attn(
                query=q, key=x, value=x,
                key_padding_mask=key_padding_mask
            )
            q = norm1(q + attn_out)
            q = norm2(q + ffn2(q))

        q = self.to_llm(q)
        # Normalize to unit std then scale to Qwen embedding distribution (std ≈ 0.02)
        q = q / (q.std(dim=-1, keepdim=True) + 1e-8)
        q = q * self.vis_scale.unsqueeze(-1)  # [B, Q, H] * [Q, 1] → per-token learned scale
        return q

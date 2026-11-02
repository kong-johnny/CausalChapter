"""Local Dependency Shift reconstructed from manuscript equations 5 and 6.

This is a new implementation, not the recovered historical checkpoint model.
Boundary class is always index 1. Ground-truth boundaries are never an input
to the dependency predictor or the boundary-logit computation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LocalDependencyShift(nn.Module):
    def __init__(self, hidden_size: int, window_size: int = 5, radius: int = 5):
        super().__init__()
        if window_size < 1 or radius < 1:
            raise ValueError("window_size and radius must be positive")
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.radius = radius
        self.predictor = nn.Sequential(
            nn.Linear(window_size * hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, window_size * hidden_size),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def predict_following(self, preceding):
        shape = preceding.shape
        return self.predictor(preceding.flatten(-2)).reshape(shape)

    def intervention_effect(self, preceding):
        """Cosine change under each zero-mask intervention; no target input."""
        full = self.predict_following(preceding)
        effects = []
        for index in range(self.window_size):
            mask = torch.ones_like(preceding)
            mask[:, index, :] = 0
            intervened = self.predict_following(preceding * mask)
            effects.append(1 - F.cosine_similarity(full, intervened, dim=-1, eps=1e-8))
        return torch.stack(effects, dim=1).mean(dim=(1, 2)), full

    def dependency_drop(self, dependency):
        count = dependency.numel()
        if count < 2:
            return torch.zeros_like(dependency)
        indices = torch.arange(count, device=dependency.device)
        distance = (indices[:, None] - indices[None, :]).abs()
        neighbors = ((distance > 0) & (distance <= self.radius)).to(dependency.dtype)
        reference = neighbors @ dependency / neighbors.sum(dim=1).clamp_min(1)
        drops = (reference - dependency).clamp_min(0)
        return (drops - drops.min()) / (drops.max() - drops.min()).clamp_min(1e-8)

    def forward(self, features):
        count = features.shape[0]
        window = self.window_size
        zero = features.sum() * 0
        if count < 2 * window:
            return features.new_zeros(count), zero
        windows = features.unfold(0, 2 * window, 1).permute(0, 2, 1)
        preceding, following = windows[:, :window], windows[:, window:]
        dependency, predicted = self.intervention_effect(preceding)
        reconstruction = F.mse_loss(predicted, following.detach())
        valid_shift = self.dependency_drop(dependency)
        # A window ending at utterance i contributes to the boundary after i.
        shift = F.pad(valid_shift, (window - 1, window))
        return shift, reconstruction


class LCDSBoundaryHead(nn.Module):
    def __init__(self, text_dim: int, visual_dim: int = 512, hidden_size: int = 256, window_size: int = 5, visual_radius=None):
        super().__init__()
        if visual_radius is not None and visual_radius < 0:
            raise ValueError("visual_radius must be nonnegative or None")
        self.visual_radius = visual_radius
        self.text_projection = nn.Linear(text_dim, hidden_size)
        self.visual_projection = nn.Linear(visual_dim, hidden_size)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)
        self.dependency = LocalDependencyShift(hidden_size, window_size)

    def forward(self, text_features, visual_features):
        text = self.text_projection(text_features)
        visual = self.visual_projection(visual_features)
        attention_mask = None
        if self.visual_radius is not None:
            indices = torch.arange(text.shape[0], device=text.device)
            attention_mask = (indices[:, None] - indices[None, :]).abs() > self.visual_radius
        attended, _ = self.cross_attention(text[None], visual[None], visual[None],
                                         attn_mask=attention_mask, need_weights=False)
        features = self.norm(text + attended[0])
        base = self.classifier(features)
        shift, reconstruction = self.dependency(features)
        bias = torch.stack((torch.zeros_like(shift), self.dependency.gamma * shift), dim=-1)
        return {"logits": base + bias, "base_logits": base, "dependency_shift": shift, "reconstruction_loss": reconstruction}

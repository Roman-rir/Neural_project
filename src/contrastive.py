"""Optional Task 4 graph-text contrastive objective."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class GraphTextContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None: super().__init__(); self.temperature = temperature
    def forward(self, graph_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        logits = F.normalize(graph_embeddings, dim=-1) @ F.normalize(text_embeddings, dim=-1).T / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2

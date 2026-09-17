"""Small Student-only Teacher-interval embedding; no PointGPT dependencies."""

import torch
from torch import nn


class StepConditionEmbedding(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, feature_dim))
        # Small nonzero projection preserves initial behavior while allowing both
        # layers to receive gradients immediately and distinguish target steps.
        nn.init.normal_(self.mlp[-1].weight, std=1e-4)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, start_step, target_step, batch_size):
        reference = self.mlp[0].weight
        values = []
        for step in (start_step, target_step):
            value = torch.as_tensor(step, device=reference.device, dtype=reference.dtype).detach().reshape(-1)
            if value.numel() == 1:
                value = value.expand(batch_size)
            if value.numel() != batch_size or not torch.isfinite(value).all():
                raise ValueError('Step condition must be a finite scalar or a vector of batch_size')
            values.append(value)
        start, target = values
        if ((start < 0) | (target > 16) | (target <= start) |
                (start != start.round()) | (target != target.round())).any():
            raise ValueError('Step condition requires integer 0 <= start_step < target_step <= 16')
        condition = torch.stack((start, target, target - start), dim=-1) / 16.0
        return self.mlp(condition)

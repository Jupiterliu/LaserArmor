from __future__ import annotations

import torch
import torch.nn as nn


class RowWiseAffineResidualModel(nn.Module):
    def __init__(
        self,
        row_hidden: int = 24,
        local_dim: int = 16,
        affine_scale: float = 0.35,
        bias_scale: float = 0.25,
        residual_scale: float = 0.20,
    ) -> None:
        super().__init__()
        self.affine_scale = float(affine_scale)
        self.bias_scale = float(bias_scale)
        self.residual_scale = float(residual_scale)
        self.row_net = nn.Sequential(
            nn.Conv1d(9, int(row_hidden), kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(int(row_hidden), int(row_hidden), kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(int(row_hidden), 6, kernel_size=1),
        )
        self.local_head = nn.Sequential(
            nn.Conv2d(3, int(local_dim), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(local_dim), int(local_dim), kernel_size=3, padding=1, groups=int(local_dim)),
            nn.GELU(),
            nn.Conv2d(int(local_dim), int(local_dim), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(local_dim), 3, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        row_features = torch.cat(
            [
                image.mean(dim=3),
                image.std(dim=3, unbiased=False),
                image.amax(dim=3),
            ],
            dim=1,
        )
        parameters = self.row_net(row_features).unsqueeze(-1)
        scale = 1.0 + self.affine_scale * torch.tanh(parameters[:, 0:3])
        bias = self.bias_scale * torch.tanh(parameters[:, 3:6])
        local = self.residual_scale * torch.tanh(self.local_head(image))
        return image * scale + bias + local


def build_residual_model(checkpoint: dict[str, object]) -> RowWiseAffineResidualModel:
    training = checkpoint.get("train_args", {})
    if not isinstance(training, dict):
        training = {}
    return RowWiseAffineResidualModel(
        row_hidden=int(training.get("row_hidden", 24)),
        local_dim=int(training.get("local_dim", 16)),
        affine_scale=float(training.get("affine_scale", 0.35)),
        bias_scale=float(training.get("bias_scale", 0.25)),
        residual_scale=float(training.get("residual_scale", 0.20)),
    )


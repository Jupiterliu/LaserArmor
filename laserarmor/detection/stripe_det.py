from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

from .row_boundaries import refine_row_spans_from_edges


Span = Tuple[int, int]


class StripeDetDetector:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        sample_columns: int = 1,
        confirm_frames: int = 1,
        minimum_stripe_height: int = 60,
        stripe_metric_mode: str = "channel_selective",
    ) -> None:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        if "stripe_thresh_T" not in checkpoint:
            raise RuntimeError(f"StripeDet checkpoint is missing stripe_thresh_T: {checkpoint_path}")
        self.threshold = float(checkpoint["stripe_thresh_T"])
        self.metric_mode = str(checkpoint.get("stripe_metric_mode", stripe_metric_mode)).strip().lower()
        if self.metric_mode not in {"luma", "channel_selective"}:
            raise RuntimeError(f"Unsupported StripeDet metric mode: {self.metric_mode}")
        self.sample_columns = max(1, int(sample_columns))
        self.confirm_frames = max(1, int(confirm_frames))
        self.minimum_stripe_height = max(1, int(minimum_stripe_height))
        self.device = device
        self.attack_streak = 0

    def _row_channel_means(self, frame: torch.Tensor) -> torch.Tensor:
        frame = frame.squeeze(0)
        columns = torch.arange(0, frame.shape[2], self.sample_columns, device=frame.device)
        return frame[:, :, columns].mean(dim=2)

    def _stripe_profile(self, frame: torch.Tensor):
        means = self._row_channel_means(frame)
        if means.shape[1] <= 1:
            empty = means.new_zeros((0,))
            return 0.0, empty, empty.to(torch.long)
        differences = means[:, 1:] - means[:, :-1]
        if self.metric_mode == "luma":
            profile = 0.114 * differences[0] + 0.587 * differences[1] + 0.299 * differences[2]
            return float(profile.abs().max()), profile, None
        magnitudes = differences.abs()
        dominant_magnitude, dominant_channel = magnitudes.max(dim=0)
        other_mean = (magnitudes.sum(dim=0) - dominant_magnitude) / 2.0
        selective = torch.clamp(dominant_magnitude - other_mean, min=0.0)
        signed = differences.gather(0, dominant_channel.unsqueeze(0)).squeeze(0)
        profile = torch.sign(signed) * selective
        score = float(selective.max()) if selective.numel() else 0.0
        return score, profile, dominant_channel

    def _spans(self, signed_edges: torch.Tensor, dominant_channels: torch.Tensor | None) -> list[Span]:
        spans: list[Span] = []
        current_start = None
        current_channel = None
        channels = dominant_channels.tolist() if dominant_channels is not None else None
        for index, value in enumerate(signed_edges.tolist()):
            row = index + 1
            channel = channels[index] if channels is not None else None
            if current_start is None:
                if value > self.threshold:
                    current_start = row
                    current_channel = channel
            else:
                same_channel = channels is None or channel == current_channel
                if value < -self.threshold and same_channel:
                    height = row - current_start
                    if height >= self.minimum_stripe_height:
                        spans.append((current_start, height))
                    current_start = None
                    current_channel = None
        if current_start is not None and channels is None:
            height = int(signed_edges.shape[0] + 1 - current_start)
            if height >= self.minimum_stripe_height:
                spans.append((current_start, height))
        return refine_row_spans_from_edges(
            signed_edges=signed_edges,
            dominant_channels=dominant_channels,
            spans=spans,
            image_height=int(signed_edges.shape[0] + 1),
            minimum_span_height=self.minimum_stripe_height,
        )

    def step(self, frame: torch.Tensor):
        frame = frame.squeeze(0)
        score, signed_edges, dominant_channels = self._stripe_profile(frame)
        spans = self._spans(signed_edges, dominant_channels) if score > self.threshold else []
        frame_flag = bool(spans)
        self.attack_streak = self.attack_streak + 1 if frame_flag else 0
        attack_flag = int(self.attack_streak >= self.confirm_frames)
        mask = np.zeros(tuple(frame.shape[1:]), dtype=np.uint8)
        for start, height in spans:
            y0 = max(0, int(start))
            y1 = min(mask.shape[0], y0 + int(height))
            mask[y0:y1] = 255
        return attack_flag, (float(score), float(score), len(spans), spans, mask)


def spans_to_mask(spans: Sequence[Span], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for start, height in spans:
        y0 = max(0, int(start))
        y1 = min(shape[0], y0 + int(height))
        mask[y0:y1] = 255
    return mask

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .blinded_by_beam import BlindedByBeamDetector
from .laser_guard import LaserGuardDetector
from .stripe_det import StripeDetDetector, spans_to_mask


@dataclass
class DetectionResult:
    attacked: bool
    score: float
    mask: np.ndarray
    spans: list[tuple[int, int]]
    details: dict[str, Any]


def image_to_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image_bgr.copy()).permute(2, 0, 1).float().div_(255.0)
    return tensor.to(device)


def build_detector(
    name: str,
    config: dict[str, Any],
    device: torch.device,
    project_root: Path,
):
    key = str(name).strip().lower()
    if key == "laserguard":
        return LaserGuardDetector(**dict(config.get("parameters", {})))
    if key == "blindedbybeam":
        artifact = project_root / str(config["artifact"])
        parameters = dict(config.get("parameters", {}))
        return BlindedByBeamDetector(artifact_path=artifact, **parameters)
    if key == "stripedet":
        checkpoint = project_root / str(config["checkpoint"])
        parameters = dict(config.get("parameters", {}))
        return StripeDetDetector(checkpoint_path=checkpoint, device=device, **parameters)
    raise RuntimeError(f"Unsupported detector: {name}")


def run_detector(detector, image_bgr: np.ndarray, device: torch.device) -> DetectionResult:
    attack_flag, payload = detector.step(image_to_tensor(image_bgr, device))
    score = float(payload[0])
    spans = [(int(start), int(height)) for start, height in payload[3]]
    mask = payload[4] if len(payload) > 4 else spans_to_mask(spans, image_bgr.shape[:2])
    details: dict[str, Any] = {}
    if len(payload) > 5:
        candidate = payload[5]
        if isinstance(candidate, dict):
            details = candidate
    elif hasattr(detector, "last_details"):
        details = dict(detector.last_details)
    return DetectionResult(
        attacked=bool(attack_flag),
        score=score,
        mask=np.asarray(mask, dtype=np.uint8),
        spans=spans,
        details=details,
    )


__all__ = [
    "BlindedByBeamDetector",
    "DetectionResult",
    "LaserGuardDetector",
    "StripeDetDetector",
    "build_detector",
    "run_detector",
]


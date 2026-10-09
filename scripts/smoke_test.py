from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from laserarmor.detection import LaserGuardDetector, StripeDetDetector, run_detector
from laserarmor.metrics import mask_overlap, psnr, ssim
from laserarmor.restoration.dominant_remove import StripeResDominantRemove
from scripts.prepare_evaluation_data import build_spans, synthesize_attack


def main() -> None:
    for height in range(1, 32):
        for style in ("single", "multi"):
            spans = build_spans(height, style, random.Random(height))
            assert spans
            assert all(start >= 0 and span_height > 0 and start + span_height <= height for start, span_height in spans)
    height, width = 128, 192
    clean = np.full((height, width, 3), 64, dtype=np.uint8)
    attacked, target_mask, spans = synthesize_attack(
        clean,
        "green",
        "single",
        random.Random(7),
        np.random.default_rng(7),
    )
    coarse, info = StripeResDominantRemove().restore(attacked, mask=target_mask, spans=spans)
    assert coarse.shape == clean.shape
    assert info["backend"] == "StripeRes-DominantRemove"
    assert psnr(clean, attacked) >= 0.0
    assert -1.0 <= ssim(clean, coarse) <= 1.0
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "stripedet.pth"
        torch.save(
            {
                "stripe_thresh_T": 0.02,
                "stripe_metric_mode": "channel_selective",
            },
            checkpoint,
        )
        detector = StripeDetDetector(checkpoint, device, minimum_stripe_height=4)
        result = run_detector(detector, attacked, device)
        assert result.mask.shape == target_mask.shape
        assert result.attacked
        overlap = mask_overlap(result.mask, target_mask)
        assert all(0.0 <= value <= 1.0 for value in overlap)
    baseline = run_detector(LaserGuardDetector(), clean, device)
    assert baseline.mask.shape == target_mask.shape
    print("LaserArmor smoke test passed")


if __name__ == "__main__":
    main()

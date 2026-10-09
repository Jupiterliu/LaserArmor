from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


Span = Tuple[int, int]
LUMA = np.asarray([0.114, 0.587, 0.299], dtype=np.float32)


def normalize_spans(image_height: int, spans: Sequence[Span]) -> list[Span]:
    intervals: list[tuple[int, int]] = []
    for start, height in spans:
        y0 = max(0, min(int(image_height), int(start)))
        y1 = max(y0, min(int(image_height), int(start + height)))
        if y1 > y0:
            intervals.append((y0, y1))
    intervals.sort()
    merged: list[list[int]] = []
    for y0, y1 in intervals:
        if not merged or y0 > merged[-1][1]:
            merged.append([y0, y1])
        else:
            merged[-1][1] = max(merged[-1][1], y1)
    return [(y0, y1 - y0) for y0, y1 in merged]


def mask_to_spans(mask: np.ndarray) -> list[Span]:
    flags = np.any(np.asarray(mask) > 0, axis=1)
    spans: list[Span] = []
    start = None
    for row, active in enumerate(flags.tolist()):
        if active and start is None:
            start = row
        elif not active and start is not None:
            spans.append((start, row - start))
            start = None
    if start is not None:
        spans.append((start, len(flags) - start))
    return spans


def row_luminance(image_bgr: np.ndarray) -> np.ndarray:
    means = np.asarray(image_bgr, dtype=np.float32).mean(axis=1)
    return means @ LUMA


def fit_reference(
    red: np.ndarray,
    blue: np.ndarray,
    green: np.ndarray,
    attacked_rows: np.ndarray,
    percentile: float,
    saturation: float,
) -> tuple[np.ndarray | None, int]:
    excess = green - 0.5 * (red + blue)
    selected = (~attacked_rows[:, None]) & (green < float(saturation))
    if not np.any(selected):
        return None, 0
    cutoff = float(np.percentile(excess[selected], float(np.clip(percentile, 0.0, 100.0))))
    selected &= excess <= cutoff
    count = int(np.count_nonzero(selected))
    if count < 3:
        return None, count
    design = np.stack(
        [red[selected], blue[selected], np.ones(count, dtype=np.float32)],
        axis=1,
    ).astype(np.float64, copy=False)
    target = green[selected].astype(np.float64, copy=False)
    try:
        coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None, count
    return coefficients.astype(np.float32), count


def depress_residual_brightness(
    image_bgr: np.ndarray,
    spans: Sequence[Span],
    edge_rows: int,
    transition_rows: int,
    strength: float,
    maximum_luma_drop: float,
    minimum_row_scale: float,
) -> np.ndarray:
    output = image_bgr.copy()
    profile = row_luminance(image_bgr)
    image_height = image_bgr.shape[0]
    for start, span_height in normalize_spans(image_height, spans):
        y0 = int(start)
        y1 = int(start + span_height)
        guard = min(max(1, int(transition_rows)), max(1, span_height // 4))
        upper0 = max(0, y0 - max(1, int(edge_rows)))
        lower1 = min(image_height, y1 + max(1, int(edge_rows)))
        inner0 = y0 + guard
        inner1 = y1 - guard
        if not (upper0 < y0 and y1 < lower1 and inner0 < inner1):
            continue
        top1 = min(inner1, inner0 + max(1, int(edge_rows)))
        bottom0 = max(inner0, inner1 - max(1, int(edge_rows)))
        top_excess = max(float(profile[inner0:top1].mean() - profile[upper0:y0].mean()), 0.0)
        bottom_excess = max(float(profile[bottom0:inner1].mean() - profile[y1:lower1].mean()), 0.0)
        shift = min(max(0.0, float(strength)) * min(top_excess, bottom_excess), max(0.0, float(maximum_luma_drop)))
        if shift <= 0.0:
            continue
        feather = np.ones(span_height, dtype=np.float32)
        transition = min(guard, span_height // 2)
        if transition > 0:
            ramp = np.arange(transition, dtype=np.float32) / float(transition)
            feather[:transition] = ramp
            feather[-transition:] = ramp[::-1]
        current = profile[y0:y1]
        target = np.maximum(current - shift * feather, 0.0)
        scale = np.clip(target / np.maximum(current, 1e-6), float(minimum_row_scale), 1.0)
        output[y0:y1] = np.clip(
            np.round(image_bgr[y0:y1].astype(np.float32) * scale[:, None, None]),
            0,
            255,
        ).astype(np.uint8)
    return output


class StripeResDominantRemove:
    def __init__(
        self,
        sigma: float = 9.0,
        fit_percentile: float = 95.0,
        saturation_threshold: float = 250.0,
        blend_start: float = 245.0,
        edge_rows: int = 5,
        transition_rows: int = 3,
        depression_strength: float = 1.0,
        maximum_luma_drop: float = 10.0,
        minimum_row_scale: float = 0.9,
    ) -> None:
        self.sigma = float(sigma)
        self.fit_percentile = float(fit_percentile)
        self.saturation_threshold = float(saturation_threshold)
        self.blend_start = float(blend_start)
        self.edge_rows = int(edge_rows)
        self.transition_rows = int(transition_rows)
        self.depression_strength = float(depression_strength)
        self.maximum_luma_drop = float(maximum_luma_drop)
        self.minimum_row_scale = float(minimum_row_scale)

    def restore(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray | None = None,
        spans: Sequence[Span] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        effective_spans = normalize_spans(
            image_bgr.shape[0],
            list(spans or mask_to_spans(mask if mask is not None else np.zeros(image_bgr.shape[:2], dtype=np.uint8))),
        )
        attacked_rows = np.zeros(image_bgr.shape[0], dtype=bool)
        for start, height in effective_spans:
            attacked_rows[start : start + height] = True
        if not effective_spans:
            return image_bgr.copy(), {"backend": "StripeRes-DominantRemove", "fallback": "empty_mask"}
        attacked_pixels = image_bgr[attacked_rows].astype(np.float32)
        blue_pixels = attacked_pixels[:, 0]
        green_pixels = attacked_pixels[:, 1]
        red_pixels = attacked_pixels[:, 2]
        channel_scores = np.asarray(
            [
                np.maximum(blue_pixels - 0.5 * (green_pixels + red_pixels), 0.0).mean(),
                np.maximum(green_pixels - 0.5 * (blue_pixels + red_pixels), 0.0).mean(),
                np.maximum(red_pixels - 0.5 * (blue_pixels + green_pixels), 0.0).mean(),
            ],
            dtype=np.float32,
        )
        target_channel = int(np.argmax(channel_scores))
        if float(channel_scores[target_channel]) <= 1e-6:
            target_channel = 1
        working = image_bgr.copy()
        if target_channel == 0:
            working[:, :, [0, 1]] = working[:, :, [1, 0]]
        elif target_channel == 2:
            working[:, :, [1, 2]] = working[:, :, [2, 1]]
        blue, green, red = cv2.split(working.astype(np.float32))
        coefficients, sample_count = fit_reference(
            red,
            blue,
            green,
            attacked_rows,
            self.fit_percentile,
            self.saturation_threshold,
        )
        if coefficients is None:
            return image_bgr.copy(), {"backend": "StripeRes-DominantRemove", "fallback": "fit_failed"}
        a, b, c = coefficients
        reference = a * red + b * blue + c
        output = working.copy()
        margin = max(1, int(np.ceil(3.0 * self.sigma)))
        alpha_width = max(1.0, self.saturation_threshold - self.blend_start)
        for start, height in effective_spans:
            y0 = int(start)
            y1 = int(start + height)
            ys = max(0, y0 - margin)
            ye = min(image_bgr.shape[0], y1 + margin)
            seed = np.zeros((ye - ys, image_bgr.shape[1]), dtype=np.float32)
            confidence = np.zeros_like(seed)
            seed[y0 - ys : y1 - ys] = np.maximum(green[y0:y1] - reference[y0:y1], 0.0)
            confidence[y0 - ys : y1 - ys] = np.clip(
                (self.saturation_threshold - green[y0:y1]) / 20.0,
                0.0,
                1.0,
            )
            small = (max(1, working.shape[1] // 2), max(1, (ye - ys) // 2))
            numerator = cv2.resize(seed * confidence, small, interpolation=cv2.INTER_AREA)
            denominator = cv2.resize(confidence, small, interpolation=cv2.INTER_AREA)
            numerator = cv2.GaussianBlur(numerator, (0, 0), self.sigma / 2.0, borderType=cv2.BORDER_CONSTANT)
            denominator = cv2.GaussianBlur(denominator, (0, 0), self.sigma / 2.0, borderType=cv2.BORDER_CONSTANT) + 1e-6
            stripe_model = cv2.resize(numerator / denominator, (working.shape[1], ye - ys), interpolation=cv2.INTER_LINEAR)
            alpha = np.clip((green[ys:ye] - self.blend_start) / alpha_width, 0.0, 1.0)
            restored = (1.0 - alpha) * (green[ys:ye] - stripe_model) + alpha * reference[ys:ye]
            output[y0:y1, :, 1] = np.clip(np.round(restored[y0 - ys : y1 - ys]), 0, 255).astype(np.uint8)
        if target_channel == 0:
            output[:, :, [0, 1]] = output[:, :, [1, 0]]
        elif target_channel == 2:
            output[:, :, [1, 2]] = output[:, :, [2, 1]]
        output = depress_residual_brightness(
            output,
            effective_spans,
            self.edge_rows,
            self.transition_rows,
            self.depression_strength,
            self.maximum_luma_drop,
            self.minimum_row_scale,
        )
        return output, {
            "backend": "StripeRes-DominantRemove",
            "fit_samples": sample_count,
            "span_count": len(effective_spans),
            "target_channel": ["blue", "green", "red"][target_channel],
        }

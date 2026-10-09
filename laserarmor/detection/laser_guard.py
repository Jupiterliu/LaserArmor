from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


Span = Tuple[int, int]


@dataclass(frozen=True)
class _CropGeometry:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    crop_top: int
    crop_left: int
    crop_height: int
    crop_width: int


@dataclass
class LaserGuardDetails:
    attack_flag: int
    detection_score: float
    spot_attack: bool
    spot_score: float
    spot_count: int
    beam_attack: bool
    beam_score: float
    beam_mask_area: int
    beam_area_score: float
    beam_feature_score: float
    beam_mean_ratio: float
    beam_std_ratio: float
    beam_mean_difference: float
    beam_std_difference: float
    beam_feature_mode: str


def _odd_kernel(value: int, label: str) -> int:
    value = int(value)
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{label} must be a positive odd integer, got {value}.")
    return value


def _frame_tensor_to_bgr_uint8(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach()
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise RuntimeError(f"Expected frame with shape (3,H,W) or (1,3,H,W), got {tuple(frame.shape)}")
    if frame.dtype.is_floating_point:
        frame = frame.clamp(0.0, 1.0).mul(255.0)
    return frame.round().byte().permute(1, 2, 0).cpu().numpy()


def _mask_to_row_spans(mask: np.ndarray) -> list[Span]:
    row_flags = np.any(mask > 0, axis=1).tolist()
    spans: list[Span] = []
    start: int | None = None
    for row, flag in enumerate(row_flags):
        if flag and start is None:
            start = row
        elif not flag and start is not None:
            spans.append((start, row - start))
            start = None
    if start is not None:
        spans.append((start, len(row_flags) - start))
    return spans


class LaserGuardDetector:
    pass

    supports_localization = True

    def __init__(
        self,
        input_size: int = 224,
        resize_short_side: int = 256,
        spot_use_standardized_view: bool = False,
        spot_autocontrast: bool = True,
        spot_grayscale_threshold: int = 140,
        spot_gaussian_kernel: int = 11,
        spot_dilate_iterations: int = 3,
        spot_component_max_area: int = 200,
        spot_count_threshold: int = 11,
        beam_solarize_threshold: int = 250,
        beam_sharpness_factor: float = 1000.0,
        beam_color_filter_threshold: int = 84,
        beam_median_kernel: int = 7,
        beam_mask_area_threshold: int = 2894,
        beam_feature_mode: str = "authors_ratio",
        beam_mean_ratio_threshold: float = 1.24,
        beam_std_ratio_threshold: float = 3.94,
        beam_mean_difference_threshold: float = 0.0,
        beam_std_difference_threshold: float = 0.0,
    ) -> None:
        self.input_size = max(1, int(input_size))
        self.resize_short_side = max(self.input_size, int(resize_short_side))
        self.spot_use_standardized_view = bool(spot_use_standardized_view)
        self.spot_autocontrast = bool(spot_autocontrast)
        self.spot_grayscale_threshold = int(np.clip(spot_grayscale_threshold, 0, 255))
        self.spot_gaussian_kernel = _odd_kernel(spot_gaussian_kernel, "spot_gaussian_kernel")
        self.spot_dilate_iterations = max(0, int(spot_dilate_iterations))
        self.spot_component_max_area = max(1, int(spot_component_max_area))
        self.spot_count_threshold = max(0, int(spot_count_threshold))
        self.beam_solarize_threshold = int(np.clip(beam_solarize_threshold, 0, 255))
        self.beam_sharpness_factor = max(0.0, float(beam_sharpness_factor))
        self.beam_color_filter_threshold = int(np.clip(beam_color_filter_threshold, 0, 127))
        self.beam_median_kernel = _odd_kernel(beam_median_kernel, "beam_median_kernel")
        self.beam_mask_area_threshold = max(0, int(beam_mask_area_threshold))
        self.beam_feature_mode = str(beam_feature_mode).strip().lower()
        if self.beam_feature_mode not in {"authors_ratio", "paper_and", "paper_either"}:
            raise ValueError(
                "beam_feature_mode must be one of authors_ratio, paper_and, or paper_either; "
                f"got {beam_feature_mode!r}."
            )
        self.beam_mean_ratio_threshold = max(1e-6, float(beam_mean_ratio_threshold))
        self.beam_std_ratio_threshold = max(1e-6, float(beam_std_ratio_threshold))
        self.beam_mean_difference_threshold = float(beam_mean_difference_threshold)
        self.beam_std_difference_threshold = float(beam_std_difference_threshold)

                                                                                    
        self.score_threshold = 1.0
        self.frame_idx = 0
        self.last_details: dict[str, object] = {}

    def _standardize(self, image_rgb: np.ndarray) -> tuple[np.ndarray, _CropGeometry]:
        original_height, original_width = image_rgb.shape[:2]
        scale = float(self.resize_short_side) / float(min(original_height, original_width))
        resized_height = max(self.input_size, int(round(original_height * scale)))
        resized_width = max(self.input_size, int(round(original_width * scale)))

        pil_image = Image.fromarray(image_rgb, mode="RGB")
        resized = pil_image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
        crop_top = max(0, (resized_height - self.input_size) // 2)
        crop_left = max(0, (resized_width - self.input_size) // 2)
        crop = resized.crop(
            (
                crop_left,
                crop_top,
                crop_left + self.input_size,
                crop_top + self.input_size,
            )
        )
        geometry = _CropGeometry(
            original_height=original_height,
            original_width=original_width,
            resized_height=resized_height,
            resized_width=resized_width,
            crop_top=crop_top,
            crop_left=crop_left,
            crop_height=self.input_size,
            crop_width=self.input_size,
        )
        return np.asarray(crop, dtype=np.uint8), geometry

    @staticmethod
    def _project_mask(mask: np.ndarray, geometry: _CropGeometry) -> np.ndarray:
        canvas = np.zeros((geometry.resized_height, geometry.resized_width), dtype=np.uint8)
        y0, x0 = geometry.crop_top, geometry.crop_left
        canvas[y0 : y0 + geometry.crop_height, x0 : x0 + geometry.crop_width] = mask
        return cv2.resize(
            canvas,
            (geometry.original_width, geometry.original_height),
            interpolation=cv2.INTER_NEAREST,
        )

    def _spot_mask(self, image_rgb: np.ndarray) -> tuple[np.ndarray, int]:
        spot_image = Image.fromarray(image_rgb, mode="RGB")
        if self.spot_autocontrast:
            spot_image = ImageOps.autocontrast(spot_image)
        spot_rgb = np.asarray(spot_image, dtype=np.uint8)
        gray = cv2.cvtColor(spot_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            (self.spot_gaussian_kernel, self.spot_gaussian_kernel),
            sigmaX=0,
        )
        _, bright = cv2.threshold(
            blurred,
            self.spot_grayscale_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        if self.spot_dilate_iterations > 0:
            bright = cv2.dilate(bright, None, iterations=self.spot_dilate_iterations)

        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            bright,
            connectivity=8,
        )
        mask = np.zeros_like(bright, dtype=np.uint8)
        selected_count = 0
        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < self.spot_component_max_area:
                mask[labels == component_id] = 255
                selected_count += 1
        return mask, selected_count

    def _beam_mask(self, image_rgb: np.ndarray) -> np.ndarray:
                                                                         
        lut = [value if value < self.beam_solarize_threshold else 0 for value in range(256)]
        beam_image = Image.fromarray(image_rgb, mode="RGB").point(lut * 3)
        beam_image = ImageOps.equalize(beam_image)
        beam_image = ImageEnhance.Sharpness(beam_image).enhance(self.beam_sharpness_factor)

        filtered_rgb = np.asarray(beam_image, dtype=np.uint8)
        average = filtered_rgb.astype(np.uint16).sum(axis=2) // 3
        distance_to_extreme = np.minimum(average, 255 - average)
        binary = np.where(distance_to_extreme < self.beam_color_filter_threshold, 255, 0).astype(np.uint8)
        binary_image = Image.fromarray(binary, mode="L").filter(ImageFilter.MedianFilter(self.beam_median_kernel))
        return np.where(np.asarray(binary_image, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)

    def _beam_statistics(self, image_rgb: np.ndarray, beam_mask: np.ndarray) -> dict[str, object]:
        mask = beam_mask > 0
        inverse = ~mask
        if not np.any(mask) or not np.any(inverse):
            return {
                "feature_pass": False,
                "feature_score": 0.0,
                "mean_ratio": 0.0,
                "std_ratio": 0.0,
                "mean_difference": 0.0,
                "std_difference": 0.0,
            }

        pixels = image_rgb.astype(np.float32, copy=False)
        inside = pixels[mask]
        outside = pixels[inverse]
        inside_mean = inside.mean(axis=0)
        outside_mean = outside.mean(axis=0)
        inside_std = inside.std(axis=0)
        outside_std = outside.std(axis=0)

        eps = 1e-6
        mean_ratios = outside_mean / np.maximum(inside_mean, eps)
        std_ratios = inside_std / np.maximum(outside_std, eps)
        mean_differences = inside_mean - outside_mean
        std_differences = inside_std - outside_std

        max_mean_ratio = float(np.max(mean_ratios))
        max_std_ratio = float(np.max(std_ratios))
        max_mean_difference = float(np.max(mean_differences))
        min_std_difference = float(np.min(std_differences))

        if self.beam_feature_mode == "authors_ratio":
            mean_scores = mean_ratios / self.beam_mean_ratio_threshold
            std_scores = std_ratios / self.beam_std_ratio_threshold
            feature_score = float(max(np.max(mean_scores), np.max(std_scores)))
            feature_pass = bool(
                np.any(mean_ratios > self.beam_mean_ratio_threshold)
                or np.any(std_ratios > self.beam_std_ratio_threshold)
            )
        else:
            contrast_scores = 1.0 + (
                mean_differences - self.beam_mean_difference_threshold
            ) / 255.0
            low_variance_scores = 1.0 + (
                self.beam_std_difference_threshold - std_differences
            ) / 255.0
            if self.beam_feature_mode == "paper_and":
                channel_scores = np.minimum(contrast_scores, low_variance_scores)
                feature_score = float(np.max(channel_scores))
                feature_pass = bool(
                    np.any(
                        (mean_differences > self.beam_mean_difference_threshold)
                        & (std_differences < self.beam_std_difference_threshold)
                    )
                )
            else:
                feature_score = float(max(np.max(contrast_scores), np.max(low_variance_scores)))
                feature_pass = bool(
                    np.any(mean_differences > self.beam_mean_difference_threshold)
                    or np.any(std_differences < self.beam_std_difference_threshold)
                )

        return {
            "feature_pass": feature_pass,
            "feature_score": max(0.0, feature_score),
            "mean_ratio": max_mean_ratio,
            "std_ratio": max_std_ratio,
            "mean_difference": max_mean_difference,
            "std_difference": min_std_difference,
        }

    def step(self, frame: torch.Tensor):
        frame_bgr = _frame_tensor_to_bgr_uint8(frame)
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        work_rgb, geometry = self._standardize(image_rgb)

        spot_input = work_rgb if self.spot_use_standardized_view else image_rgb
        spot_mask, spot_count = self._spot_mask(spot_input)
        spot_score = float(spot_count) / float(max(1, self.spot_count_threshold))
        spot_attack = bool(spot_count > self.spot_count_threshold)

        beam_mask = self._beam_mask(work_rgb)
        beam_mask_area = int(np.count_nonzero(beam_mask))
        beam_area_score = float(beam_mask_area) / float(max(1, self.beam_mask_area_threshold))
        beam_stats = self._beam_statistics(work_rgb, beam_mask)
        beam_feature_score = float(beam_stats["feature_score"])
        beam_score = min(beam_area_score, beam_feature_score)
        beam_attack = bool(
            beam_mask_area > self.beam_mask_area_threshold
            and bool(beam_stats["feature_pass"])
        )

        detector_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        if spot_attack:
            projected_spot_mask = (
                self._project_mask(spot_mask, geometry)
                if self.spot_use_standardized_view
                else spot_mask
            )
            detector_mask = cv2.bitwise_or(detector_mask, projected_spot_mask)
        if beam_attack:
            detector_mask = cv2.bitwise_or(detector_mask, self._project_mask(beam_mask, geometry))

        attack_flag = int(spot_attack or beam_attack)
        detection_score = max(spot_score, beam_score)
        stripe_spans = _mask_to_row_spans(detector_mask)
        details = LaserGuardDetails(
            attack_flag=attack_flag,
            detection_score=float(detection_score),
            spot_attack=spot_attack,
            spot_score=float(spot_score),
            spot_count=int(spot_count),
            beam_attack=beam_attack,
            beam_score=float(beam_score),
            beam_mask_area=beam_mask_area,
            beam_area_score=float(beam_area_score),
            beam_feature_score=beam_feature_score,
            beam_mean_ratio=float(beam_stats["mean_ratio"]),
            beam_std_ratio=float(beam_stats["std_ratio"]),
            beam_mean_difference=float(beam_stats["mean_difference"]),
            beam_std_difference=float(beam_stats["std_difference"]),
            beam_feature_mode=self.beam_feature_mode,
        )
        self.last_details = asdict(details)
        self.frame_idx += 1

        return attack_flag, (
            float(detection_score),
            float(detection_score),
            len(stripe_spans),
            stripe_spans,
            detector_mask,
        )

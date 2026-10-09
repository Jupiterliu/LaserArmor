from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import cv2
import joblib
import numpy as np
import torch
from shapely.geometry import Polygon


Span = Tuple[int, int]
STRICT_FEATURE_SCHEMA = "equations_1_3_v1"
LEGACY_FEATURE_SCHEMA = "legacy_physical_v0"


@dataclass(frozen=True)
class AdvLBConfig:
    pass

    feature_schema: str = STRICT_FEATURE_SCHEMA
                                                                     
    sharpen_kernel: tuple[tuple[float, ...], ...] = (
        (0.0, -1.0, 0.0),
        (-1.0, 5.0, -1.0),
        (0.0, -1.0, 0.0),
    )
                                                                         
    laplacian_ksize: int = 3
    border_type: int = cv2.BORDER_REFLECT_101
                                               
    resize_long_side: int = 0


@dataclass(frozen=True)
class AdvLSConfig:
    pass

                                                                
    hsv_lower: tuple[int, int, int] = (58, 204, 219)
    hsv_upper: tuple[int, int, int] = (101, 255, 255)
                                         
    morphology_kernel_size: int = 5
                                                                    
    morphology_iterations: int = 1
    min_contour_area: float = 80.0
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    min_circularity: float = 0.7
    required_quadrants: int = 2


@dataclass(frozen=True)
class SpotAnalysis:
    pass

    valid_quadrants: int
    has_global_local_intersection: bool
    total_intersecting_area: float
    detector_mask: np.ndarray


def _resize_long_side(image_bgr: np.ndarray, target: int) -> np.ndarray:
    pass

    if target <= 0:
        return image_bgr
    height, width = image_bgr.shape[:2]
    scale = float(target) / float(max(height, width))
    if abs(scale - 1.0) < 1e-12:
        return image_bgr
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image_bgr, size, interpolation=interpolation)


def _validate_bgr_uint8(image_bgr: np.ndarray) -> np.ndarray:
    pass

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape (H,W,3), got {image.shape}")
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        upper = float(np.nanmax(image)) if image.size else 0.0
        scale = 255.0 if upper <= 1.0 else 1.0
        return np.clip(np.nan_to_num(image) * scale, 0.0, 255.0).round().astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def extract_blinded_by_beam_features(
    image_bgr: np.ndarray,
    resize_to: int = 0,
    config: AdvLBConfig | None = None,
) -> np.ndarray:
    pass

    cfg = config or AdvLBConfig(resize_long_side=int(resize_to))
    image = _resize_long_side(_validate_bgr_uint8(image_bgr), cfg.resize_long_side)
    image_f32 = image.astype(np.float32)                      
    kernel = np.asarray(cfg.sharpen_kernel, dtype=np.float32)         

    sharpened = cv2.filter2D(
        image_f32,
        cv2.CV_32F,
        kernel,
        borderType=cfg.border_type,
    )                      
    laplacian = cv2.Laplacian(
        image_f32,
        cv2.CV_32F,
        ksize=cfg.laplacian_ksize,
        borderType=cfg.border_type,
    )                      
    squared_difference = np.square(sharpened - laplacian)                      
    distances_bgr = np.sqrt(
        np.sum(squared_difference, axis=(0, 1), dtype=np.float64)
    )                   
                                                                        
    distance_b, distance_g, distance_r = (float(value) for value in distances_bgr)
                                                                           
    absdiff_b_r = abs(distance_b - distance_r)
    absdiff_g_r = abs(distance_g - distance_r)
    return np.asarray(
        [distance_r, distance_g, distance_b, absdiff_b_r, absdiff_g_r],
        dtype=np.float64,
    )                                


def _extract_legacy_physical_features(image_bgr: np.ndarray, resize_to: int) -> np.ndarray:
    pass

    image = _resize_long_side(_validate_bgr_uint8(image_bgr), resize_to).astype(np.float32)
    kernel = np.asarray(
        ((-1.0, -1.0, -1.0), (-1.0, 9.0, -1.0), (-1.0, -1.0, -1.0)),
        dtype=np.float32,
    )
    distances = []
    for channel_index in range(3):
        channel = image[:, :, channel_index]
        sharpened = cv2.filter2D(channel, cv2.CV_32F, kernel)
        laplacian = cv2.Laplacian(channel, cv2.CV_32F, ksize=3)
        distances.append(float(np.linalg.norm(sharpened - laplacian) / np.sqrt(channel.size)))
    blue, green, red = cv2.split(image)
    return np.asarray(
        [
            *distances,
            float(np.mean(np.abs(blue - red))),
            float(np.mean(np.abs(green - red))),
        ],
        dtype=np.float64,
    )


def _morphology_kernel(config: AdvLSConfig) -> np.ndarray:
    pass

    size = int(config.morphology_kernel_size)
    if size <= 0:
        raise ValueError("morphology_kernel_size must be positive")
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def _segment_spots(image_bgr: np.ndarray, config: AdvLSConfig) -> np.ndarray:
    pass

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)                      
    mask = cv2.inRange(
        hsv,
        np.asarray(config.hsv_lower, dtype=np.uint8),
        np.asarray(config.hsv_upper, dtype=np.uint8),
    )                    
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        _morphology_kernel(config),
        iterations=int(config.morphology_iterations),
    )                  


def _valid_spot_contours(mask: np.ndarray, config: AdvLSConfig) -> list[np.ndarray]:
    pass

                                                                 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid: list[np.ndarray] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= config.min_contour_area:
            continue
        _x, _y, width, height = cv2.boundingRect(contour)
        if height <= 0:
            continue
        aspect_ratio = float(width) / float(height)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0.0 else 0.0
        if (
            config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio
            and circularity >= config.min_circularity
        ):
            valid.append(contour)
    return valid


def _offset_contour(contour: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
    pass

    offset = np.asarray([[[x_offset, y_offset]]], dtype=contour.dtype)
    return contour + offset                      


def _contour_polygon(contour: np.ndarray) -> Polygon | None:
    pass

    points = np.asarray(contour).reshape(-1, 2)                    
    if points.shape[0] < 3:
        return None
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon if not polygon.is_empty and polygon.area > 0.0 else None


def _positive_polygon_intersection(left: np.ndarray, right: np.ndarray) -> bool:
    pass

    left_polygon = _contour_polygon(left)
    right_polygon = _contour_polygon(right)
    if left_polygon is None or right_polygon is None:
        return False
    intersection = left_polygon.intersection(right_polygon)
    return not intersection.is_empty and float(intersection.area) > 0.0


def analyze_adv_ls(image_bgr: np.ndarray, config: AdvLSConfig | None = None) -> SpotAnalysis:
    pass

    cfg = config or AdvLSConfig()
    image = _validate_bgr_uint8(image_bgr)
    global_mask = _segment_spots(image, cfg)                    
    global_contours = _valid_spot_contours(global_mask, cfg)
    height, width = global_mask.shape
    x_mid, y_mid = width // 2, height // 2
    quadrants = (
        (0, 0, x_mid, y_mid),
        (x_mid, 0, width, y_mid),
        (0, y_mid, x_mid, height),
        (x_mid, y_mid, width, height),
    )

                                                                                
    valid_quadrants = 0
    local_contours: list[np.ndarray] = []
    for x0, y0, x1, y1 in quadrants:
        local_image = image[y0:y1, x0:x1]                        
        local_mask = _segment_spots(local_image, cfg)                        
        local_valid = _valid_spot_contours(local_mask, cfg)
        if local_valid:
            valid_quadrants += 1
            local_contours.extend(_offset_contour(contour, x0, y0) for contour in local_valid)

                                                                                
    intersecting_global = [
        global_contour
        for global_contour in global_contours
        if any(
            _positive_polygon_intersection(global_contour, local_contour)
            for local_contour in local_contours
        )
    ]
    total_area = float(sum(cv2.contourArea(contour) for contour in intersecting_global))
    detector_mask = np.zeros_like(global_mask)
    if intersecting_global:
        cv2.drawContours(detector_mask, intersecting_global, -1, 255, thickness=cv2.FILLED)
    return SpotAnalysis(
        valid_quadrants=valid_quadrants,
        has_global_local_intersection=bool(intersecting_global),
        total_intersecting_area=total_area,
        detector_mask=detector_mask,
    )


def empirical_percentile(value: float, sorted_reference: np.ndarray) -> float:
    pass

    reference = np.asarray(sorted_reference, dtype=np.float64).reshape(-1)
    if reference.size == 0:
        return 0.0
    left = int(np.searchsorted(reference, value, side="left"))
    right = int(np.searchsorted(reference, value, side="right"))
    return float((left + right) / (2.0 * reference.size))


def frame_tensor_to_bgr_uint8(frame: torch.Tensor) -> np.ndarray:
    pass

    value = frame.detach()
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {tuple(value.shape)}")
        value = value[0]                        
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"Expected frame shape (3,H,W), got {tuple(value.shape)}")
    if value.dtype.is_floating_point:
        value = value.clamp(0.0, 1.0).mul(255.0)
    return value.round().byte().permute(1, 2, 0).cpu().numpy()                      


def _adv_lb_config_from_artifact(metadata: Mapping[str, Any], resize_override: int | None) -> AdvLBConfig:
    pass

    raw = metadata.get("adv_lb_config", {})
    raw = raw if isinstance(raw, Mapping) else {}
    kernel = raw.get("sharpen_kernel", AdvLBConfig.sharpen_kernel)
    resize_long_side = int(raw.get("resize_long_side", 0))
    if resize_override is not None:
        resize_long_side = int(resize_override)
    border_name = str(raw.get("border_type", "reflect_101")).lower()
    border_type = cv2.BORDER_REFLECT_101 if border_name == "reflect_101" else cv2.BORDER_DEFAULT
    return AdvLBConfig(
        feature_schema=str(metadata.get("feature_schema", raw.get("feature_schema", LEGACY_FEATURE_SCHEMA))),
        sharpen_kernel=tuple(tuple(float(value) for value in row) for row in kernel),
        laplacian_ksize=int(raw.get("laplacian_ksize", 3)),
        border_type=border_type,
        resize_long_side=resize_long_side,
    )


class BlindedByBeamDetector:
    pass

    supports_localization = True
    frame_score_source = "image_score"

    def __init__(
        self,
        artifact_path: str | Path,
        resize_to: int | None = None,
        hsv_lower: Sequence[int] = (58, 204, 219),
        hsv_upper: Sequence[int] = (101, 255, 255),
        min_contour_area: float = 80.0,
        min_total_area: float | None = None,
        required_quadrants: int = 2,
    ) -> None:
        artifact_file = Path(artifact_path).resolve()
        if not artifact_file.is_file():
            raise RuntimeError(f"Blinded-by-Beam artifact was not found: {artifact_file}")
        artifact = joblib.load(artifact_file)
        required = {"scaler", "pca", "isolation_forest"}
        missing = required - set(artifact)
        if missing:
            raise RuntimeError(f"Blinded-by-Beam artifact is missing keys: {sorted(missing)}")

        self.scaler = artifact["scaler"]
        self.pca = artifact["pca"]
        self.isolation_forest = artifact["isolation_forest"]
        self.artifact_metadata = dict(artifact.get("metadata", {}))
        self.artifact_path = str(artifact_file)
        self.adv_lb_config = _adv_lb_config_from_artifact(self.artifact_metadata, resize_to)
        self.feature_schema = self.adv_lb_config.feature_schema
        self.adv_ls_config = AdvLSConfig(
            hsv_lower=tuple(int(value) for value in hsv_lower),
            hsv_upper=tuple(int(value) for value in hsv_upper),
            min_contour_area=float(min_contour_area),
            required_quadrants=int(required_quadrants),
        )
        artifact_tau = float(self.artifact_metadata.get("adv_ls_area_threshold", 80.0))
        self.adv_ls_area_threshold = artifact_tau if min_total_area is None else float(min_total_area)
        self.clean_adv_lb_scores = np.sort(
            np.asarray(artifact.get("clean_adv_lb_scores", []), dtype=np.float64)
        )
        self.clean_adv_ls_areas = np.sort(
            np.asarray(artifact.get("clean_adv_ls_areas", []), dtype=np.float64)
        )

    def _extract_features(self, image_bgr: np.ndarray) -> np.ndarray:
        pass

        if self.feature_schema == LEGACY_FEATURE_SCHEMA:
            return _extract_legacy_physical_features(
                image_bgr,
                resize_to=self.adv_lb_config.resize_long_side or 640,
            )
        if self.feature_schema != STRICT_FEATURE_SCHEMA:
            raise RuntimeError(f"Unsupported Blinded-by-Beam feature schema: {self.feature_schema}")
        return extract_blinded_by_beam_features(image_bgr, config=self.adv_lb_config)

    def _adv_lb_branch(self, image_bgr: np.ndarray) -> tuple[bool, float, float, np.ndarray]:
        pass

        features = self._extract_features(image_bgr)[None, :]                 
        standardized = self.scaler.transform(features)                  
        projected = self.pca.transform(standardized)                  
        decision = float(self.isolation_forest.decision_function(projected)[0])
        anomaly_value = -decision
        if self.clean_adv_lb_scores.size:
            score = empirical_percentile(anomaly_value, self.clean_adv_lb_scores)
        else:
                                                                             
            scale = float(self.artifact_metadata.get("clean_decision_scale", 1.0))
            score = float(max(0.0, anomaly_value) / max(scale, 1e-12))
        return decision < 0.0, score, decision, features[0]

    def _adv_ls_branch(self, image_bgr: np.ndarray) -> tuple[bool, float, SpotAnalysis]:
        pass

        analysis = analyze_adv_ls(image_bgr, self.adv_ls_config)
        attacked = (
            analysis.valid_quadrants >= self.adv_ls_config.required_quadrants
            and analysis.has_global_local_intersection
            and analysis.total_intersecting_area > self.adv_ls_area_threshold
        )
        if self.clean_adv_ls_areas.size:
            score = empirical_percentile(
                analysis.total_intersecting_area,
                self.clean_adv_ls_areas,
            )
        else:
            score = float(
                analysis.total_intersecting_area / max(self.adv_ls_area_threshold, 1.0)
            )
        return attacked, score, analysis

    def step(self, frame: torch.Tensor):
        pass

        image_bgr = frame_tensor_to_bgr_uint8(frame)
        adv_lb_attack, adv_lb_score, adv_lb_decision, features = self._adv_lb_branch(image_bgr)
        adv_ls_attack, adv_ls_score, spot = self._adv_ls_branch(image_bgr)
        attack_flag = int(adv_lb_attack or adv_ls_attack)
                                                                             
                                                                              
        image_score = float(max(adv_lb_score, adv_ls_score))
        stripe_spans = _mask_to_row_spans(spot.detector_mask)
        details = {
            "adv_lb_attack": bool(adv_lb_attack),
            "adv_lb_score": float(adv_lb_score),
            "adv_lb_decision_function": float(adv_lb_decision),
            "adv_lb_features": features.tolist(),
            "adv_ls_attack": bool(adv_ls_attack),
            "adv_ls_score": float(adv_ls_score),
            "adv_ls_valid_quadrants": int(spot.valid_quadrants),
            "adv_ls_has_intersection": bool(spot.has_global_local_intersection),
            "adv_ls_total_area": float(spot.total_intersecting_area),
            "adv_ls_area_threshold": float(self.adv_ls_area_threshold),
        }
        return attack_flag, (
            image_score,
            image_score,
            len(stripe_spans),
            stripe_spans,
            spot.detector_mask,
            details,
        )


def _mask_to_row_spans(mask: np.ndarray) -> list[Span]:
    pass

    flags = np.any(mask > 0, axis=1)
    spans: list[Span] = []
    start: int | None = None
    for row, active in enumerate(flags):
        if active and start is None:
            start = row
        elif not active and start is not None:
            spans.append((start, row - start))
            start = None
    if start is not None:
        spans.append((start, len(flags) - start))
    return spans

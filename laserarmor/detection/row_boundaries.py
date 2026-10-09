from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


Span = tuple[int, int]


@dataclass(frozen=True)
class RowBoundaryRefinementConfig:
    pass

    minimum_edge_magnitude: float = 0.01
    noise_mad_scale: float = 8.0
    maximum_edge_cluster_gap: int = 2
    search_fraction: float = 0.40
    minimum_search_rows: int = 12
    maximum_search_rows: int = 64
    minimum_row_iou: float = 0.50
    minimum_coarse_coverage: float = 1.01


DEFAULT_ROW_BOUNDARY_CONFIG = RowBoundaryRefinementConfig()


def _frame_to_bgr_float(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach().squeeze(0).to(torch.float32).cpu()
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise RuntimeError(
                "Expected a BGR tensor with shape (3,H,W), "
                f"got {tuple(tensor.shape)}"
            )
        image = tensor.permute(1, 2, 0).numpy()
    else:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(
                "Expected a BGR image with shape (H,W,3), "
                f"got {tuple(image.shape)}"
            )
        image = image.astype(np.float32, copy=False)
        if image.dtype == np.uint8 or float(np.nanmax(image)) > 1.5:
            image = image / 255.0
    return np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)


def compute_selective_row_edges(
    frame: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pass

    image = _frame_to_bgr_float(frame)
    row_channel_means = image.mean(axis=1).T
    if row_channel_means.shape[1] <= 1:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    differences = np.diff(row_channel_means, axis=1)
    absolute = np.abs(differences)
    dominant_channel = np.argmax(absolute, axis=0).astype(np.int64)
    dominant_magnitude = np.max(absolute, axis=0)
    other_mean = (
        np.sum(absolute, axis=0) - dominant_magnitude
    ) / 2.0
    selective_magnitude = np.maximum(
        dominant_magnitude - other_mean,
        0.0,
    )
    column_indices = np.arange(differences.shape[1])
    dominant_signed = differences[dominant_channel, column_indices]
    signed_edges = (
        np.sign(dominant_signed) * selective_magnitude
    ).astype(np.float32)
    return signed_edges, dominant_channel


def _as_numpy_1d(
    values: torch.Tensor | np.ndarray | Sequence[float],
    dtype: np.dtype,
) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    return np.asarray(array, dtype=dtype).reshape(-1)


def _adaptive_edge_threshold(
    signed_edges: np.ndarray,
    config: RowBoundaryRefinementConfig,
) -> float:
    magnitude = np.abs(signed_edges.astype(np.float32, copy=False))
    if magnitude.size == 0:
        return float(config.minimum_edge_magnitude)
    median = float(np.median(magnitude))
    mad = float(np.median(np.abs(magnitude - median)))
    robust_sigma = 1.4826 * mad
    return max(
        float(config.minimum_edge_magnitude),
        median + float(config.noise_mad_scale) * robust_sigma,
    )


def _cluster_candidate_indices(
    indices: np.ndarray,
    row_numbers: np.ndarray,
    maximum_gap: int,
) -> list[list[int]]:
    if indices.size == 0:
        return []
    clusters: list[list[int]] = []
    current = [int(indices[0])]
    for raw_index in indices[1:]:
        index = int(raw_index)
        previous = current[-1]
        if int(row_numbers[index] - row_numbers[previous]) <= maximum_gap:
            current.append(index)
        else:
            clusters.append(current)
            current = [index]
    clusters.append(current)
    return clusters


def normalize_row_spans(
    image_height: int,
    spans: Sequence[Span],
) -> list[Span]:
    height = max(0, int(image_height))
    intervals: list[tuple[int, int]] = []
    for start, span_height in spans:
        y0 = max(0, min(height, int(start)))
        y1 = max(y0, min(height, int(start + span_height)))
        if y1 > y0:
            intervals.append((y0, y1))
    if not intervals:
        return []

    intervals.sort()
    merged: list[list[int]] = []
    for y0, y1 in intervals:
        if not merged or y0 > merged[-1][1]:
            merged.append([y0, y1])
        else:
            merged[-1][1] = max(merged[-1][1], y1)
    return [(y0, y1 - y0) for y0, y1 in merged]


def _row_iou(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> float:
    intersection = max(
        0,
        min(first_end, second_end) - max(first_start, second_start),
    )
    union = max(first_end, second_end) - min(first_start, second_start)
    return float(intersection) / max(float(union), 1.0)


def refine_row_spans_from_edges(
    signed_edges: torch.Tensor | np.ndarray | Sequence[float],
    dominant_channels: torch.Tensor | np.ndarray | Sequence[int] | None,
    spans: Sequence[Span],
    image_height: int | None = None,
    minimum_span_height: int = 1,
    config: RowBoundaryRefinementConfig = DEFAULT_ROW_BOUNDARY_CONFIG,
    require_boundary_pair: bool = False,
) -> list[Span]:
    pass

    edges = _as_numpy_1d(signed_edges, np.float32)
    if image_height is None:
        height = int(edges.size + 1)
    else:
        height = max(0, int(image_height))
    coarse_spans = normalize_row_spans(height, spans)
    if not coarse_spans or edges.size == 0:
        return coarse_spans

    if dominant_channels is None:
        channels = np.zeros(edges.shape, dtype=np.int64)
    else:
        channels = _as_numpy_1d(dominant_channels, np.int64)
        if channels.shape != edges.shape:
            raise RuntimeError(
                "dominant_channels and signed_edges must have the same shape, "
                f"got {channels.shape} and {edges.shape}"
            )

    threshold = _adaptive_edge_threshold(edges, config)
    edge_rows = np.arange(1, edges.size + 1, dtype=np.int32)
    minimum_height = max(1, int(minimum_span_height))
    refined: list[Span] = []

    for coarse_start, coarse_height in coarse_spans:
        coarse_end = int(coarse_start + coarse_height)
        midpoint = coarse_start + coarse_height // 2
        search_rows = int(
            np.ceil(float(coarse_height) * float(config.search_fraction))
        )
        search_rows = max(int(config.minimum_search_rows), search_rows)
        search_rows = min(int(config.maximum_search_rows), search_rows)

        positive_min = max(1, coarse_start - search_rows)
        positive_max = min(height - 1, coarse_start + search_rows)
        negative_min = max(1, coarse_end - search_rows)
        negative_max = min(height - 1, coarse_end + search_rows)
        best: tuple[float, int, int] | None = None

        for channel in np.unique(channels):
            positive_indices = np.where(
                (channels == channel)
                & (edges >= threshold)
                & (edge_rows >= positive_min)
                & (edge_rows <= positive_max)
            )[0]
            negative_indices = np.where(
                (channels == channel)
                & (edges <= -threshold)
                & (edge_rows >= negative_min)
                & (edge_rows <= negative_max)
            )[0]
            positive_clusters = _cluster_candidate_indices(
                positive_indices,
                edge_rows,
                max(0, int(config.maximum_edge_cluster_gap)),
            )
            negative_clusters = _cluster_candidate_indices(
                negative_indices,
                edge_rows,
                max(0, int(config.maximum_edge_cluster_gap)),
            )

            for positive_cluster in positive_clusters:
                peak_index = max(
                    positive_cluster,
                    key=lambda index: float(edges[index]),
                )
                refined_start = int(edge_rows[peak_index])
                positive_peak = float(edges[peak_index])

                for negative_cluster in negative_clusters:
                    refined_end = int(edge_rows[negative_cluster[-1]])
                    negative_peak = max(
                        float(-edges[index])
                        for index in negative_cluster
                    )
                    refined_height = refined_end - refined_start
                    if refined_height < minimum_height:
                        continue
                    if not (refined_start <= midpoint < refined_end):
                        continue

                    overlap = _row_iou(
                        coarse_start,
                        coarse_end,
                        refined_start,
                        refined_end,
                    )
                    coarse_intersection = max(
                        0,
                        min(coarse_end, refined_end)
                        - max(coarse_start, refined_start),
                    )
                    coarse_coverage = (
                        float(coarse_intersection)
                        / max(float(coarse_height), 1.0)
                    )
                    if (
                        overlap < float(config.minimum_row_iou)
                        and coarse_coverage
                        < float(config.minimum_coarse_coverage)
                    ):
                        continue

                    normalized_shift = (
                        abs(refined_start - coarse_start)
                        + abs(refined_end - coarse_end)
                    ) / max(float(search_rows), 1.0)
                    strength = float(
                        np.sqrt(max(positive_peak * negative_peak, 0.0))
                    )
                    score = (
                        strength * (0.75 + 0.25 * overlap)
                        / (1.0 + 0.35 * normalized_shift)
                    )
                    if best is None or score > best[0]:
                        best = (score, refined_start, refined_end)

        if best is None and not require_boundary_pair:
            refined.append((coarse_start, coarse_height))
        elif best is not None:
            refined.append((best[1], best[2] - best[1]))

    return normalize_row_spans(height, refined)


def refine_laser_row_spans(
    frame: torch.Tensor | np.ndarray,
    spans: Sequence[Span],
    minimum_span_height: int = 1,
    config: RowBoundaryRefinementConfig = DEFAULT_ROW_BOUNDARY_CONFIG,
    require_boundary_pair: bool = False,
) -> list[Span]:
    pass

    image = _frame_to_bgr_float(frame)
    signed_edges, dominant_channels = compute_selective_row_edges(image)
    return refine_row_spans_from_edges(
        signed_edges=signed_edges,
        dominant_channels=dominant_channels,
        spans=spans,
        image_height=image.shape[0],
        minimum_span_height=minimum_span_height,
        config=config,
        require_boundary_pair=require_boundary_pair,
    )

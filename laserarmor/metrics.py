from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from skimage.metrics import structural_similarity


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    error = np.mean((reference.astype(np.float64) - prediction.astype(np.float64)) ** 2)
    if error <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(255.0 / math.sqrt(error)))


def ssim(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(structural_similarity(reference, prediction, channel_axis=2, data_range=255))


def masked_psnr(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask) > 0
    if not np.any(selected):
        return float("nan")
    difference = reference.astype(np.float64)[selected] - prediction.astype(np.float64)[selected]
    error = float(np.mean(difference ** 2))
    if error <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(255.0 / math.sqrt(error)))


def masked_mae(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask) > 0
    if not np.any(selected):
        return float("nan")
    difference = np.abs(reference.astype(np.float64)[selected] - prediction.astype(np.float64)[selected])
    return float(np.mean(difference))


def mask_overlap(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    left = np.asarray(prediction) > 0
    right = np.asarray(target) > 0
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    total = int(np.count_nonzero(left) + np.count_nonzero(right))
    iou = float(intersection / union) if union else 1.0
    dice = float(2 * intersection / total) if total else 1.0
    return iou, dice


def classification_metrics(labels: list[int], predictions: list[int], scores: list[float]) -> dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    result["auroc"] = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    return result


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    valid = array[~np.isnan(array)]
    return float(valid.mean()) if valid.size else float("nan")

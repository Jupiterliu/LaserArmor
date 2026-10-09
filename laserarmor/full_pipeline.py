from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .detection import build_detector, run_detector
from .metrics import (
    classification_metrics,
    finite_mean,
    mask_overlap,
    masked_mae,
    masked_psnr,
    psnr,
    ssim,
)
from .restoration import build_restorer


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, ensure_ascii=True, allow_nan=False)


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def load_manifest(dataset_dir: Path, limit: int = 0) -> list[dict[str, Any]]:
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise RuntimeError(f"Dataset manifest was not found: {manifest_path}")
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    if not records:
        raise RuntimeError(f"Dataset manifest is empty: {manifest_path}")
    return records


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("runtime", {}).get("device", "auto")).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return torch.device(requested)


def merge_method_config(
    config: dict[str, Any],
    category: str,
    method: str,
    dataset_name: str,
) -> dict[str, Any]:
    methods = config.get("methods", {}).get(category, {})
    if method not in methods:
        raise RuntimeError(f"Missing configuration for {category} method {method}")
    base = dict(methods[method])
    overrides = base.pop("by_dataset", {})
    selected = overrides.get(dataset_name, {}) if isinstance(overrides, dict) else {}
    merged = {**base, **dict(selected)}
    if "parameters" in base or "parameters" in selected:
        merged["parameters"] = {**dict(base.get("parameters", {})), **dict(selected.get("parameters", {}))}
    return merged


def safe_id(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)


def run_evaluation(
    config: dict[str, Any],
    dataset_name: str,
    dataset_dir: Path,
    detector_name: str,
    restorer_name: str,
    output_dir: Path,
    limit: int = 0,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[1]
    device = resolve_device(config)
    detector_config = merge_method_config(config, "detectors", detector_name, dataset_name)
    restorer_config = merge_method_config(config, "restorers", restorer_name, dataset_name)
    detector = build_detector(detector_name, detector_config, device, root)
    restorer = build_restorer(restorer_name, restorer_config, device, root)
    records = load_manifest(dataset_dir, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image: list[dict[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    ious: list[float] = []
    dices: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    masked_psnrs: list[float] = []
    masked_maes: list[float] = []
    attacked_psnrs: list[float] = []
    detection_times: list[float] = []
    restoration_times: list[float] = []
    for record in records:
        identifier = safe_id(str(record["id"]))
        clean = read_image(dataset_dir / str(record["clean"]))
        attacked = read_image(dataset_dir / str(record["attacked"]))
        target_mask = read_image(dataset_dir / str(record["mask"]), grayscale=True)
        if clean.shape != attacked.shape or clean.shape[:2] != target_mask.shape:
            raise RuntimeError(f"Shape mismatch for sample {identifier}")
        clean_start = time.perf_counter()
        clean_detection = run_detector(detector, clean, device)
        clean_elapsed = (time.perf_counter() - clean_start) * 1000.0
        attack_start = time.perf_counter()
        attack_detection = run_detector(detector, attacked, device)
        attack_elapsed = (time.perf_counter() - attack_start) * 1000.0
        labels.extend([0, 1])
        predictions.extend([int(clean_detection.attacked), int(attack_detection.attacked)])
        scores.extend([clean_detection.score, attack_detection.score])
        detection_times.extend([clean_elapsed, attack_elapsed])
        overlap_iou, overlap_dice = mask_overlap(attack_detection.mask, target_mask)
        ious.append(overlap_iou)
        dices.append(overlap_dice)
        if attack_detection.attacked:
            restore_start = time.perf_counter()
            restored, restore_info = restorer.restore(
                attacked,
                mask=attack_detection.mask,
                spans=attack_detection.spans,
            )
            restore_elapsed = (time.perf_counter() - restore_start) * 1000.0
        else:
            restored = attacked.copy()
            restore_info = {"fallback": "not_detected"}
            restore_elapsed = 0.0
        restoration_times.append(restore_elapsed)
        image_psnr = psnr(clean, restored)
        image_ssim = ssim(clean, restored)
        image_masked_psnr = masked_psnr(clean, restored, target_mask)
        image_masked_mae = masked_mae(clean, restored, target_mask)
        attacked_psnr = psnr(clean, attacked)
        psnrs.append(image_psnr)
        ssims.append(image_ssim)
        masked_psnrs.append(image_masked_psnr)
        masked_maes.append(image_masked_mae)
        attacked_psnrs.append(attacked_psnr)
        restored_path = output_dir / "restored" / f"{identifier}.png"
        predicted_mask_path = output_dir / "detector_masks" / f"{identifier}.png"
        write_image(restored_path, restored)
        write_image(predicted_mask_path, attack_detection.mask)
        per_image.append(
            {
                "id": identifier,
                "attack_color": str(record.get("attack_color", "unknown")),
                "attack_style": str(record.get("attack_style", "unknown")),
                "clean_prediction": int(clean_detection.attacked),
                "clean_score": clean_detection.score,
                "attack_prediction": int(attack_detection.attacked),
                "attack_score": attack_detection.score,
                "mask_iou": overlap_iou,
                "mask_dice": overlap_dice,
                "attacked_psnr": attacked_psnr,
                "restored_psnr": image_psnr,
                "restored_ssim": image_ssim,
                "masked_psnr": image_masked_psnr,
                "masked_mae": image_masked_mae,
                "detection_ms": attack_elapsed,
                "restoration_ms": restore_elapsed,
                "restore_info": json.dumps(restore_info, ensure_ascii=True, default=str),
                "restored_path": str(restored_path.relative_to(output_dir)),
                "detector_mask_path": str(predicted_mask_path.relative_to(output_dir)),
            }
        )
    detection_summary = classification_metrics(labels, predictions, scores)
    summary = {
        "dataset": dataset_name,
        "dataset_variant": dataset_dir.name,
        "detector": detector_name,
        "restorer": restorer_name,
        "sample_count": len(records),
        "detection": {
            **detection_summary,
            "mask_iou": finite_mean(ious),
            "mask_dice": finite_mean(dices),
            "mean_latency_ms": finite_mean(detection_times),
        },
        "restoration": {
            "attacked_psnr": finite_mean(attacked_psnrs),
            "psnr": finite_mean(psnrs),
            "ssim": finite_mean(ssims),
            "masked_psnr": finite_mean(masked_psnrs),
            "masked_mae": finite_mean(masked_maes),
            "mean_latency_ms": finite_mean(restoration_times),
        },
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one LaserArmor detection and restoration pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_pipeline.json"))
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--detector", choices=["LaserGuard", "BlindedByBeam", "StripeDet"], required=True)
    parser.add_argument(
        "--restorer",
        choices=[
            "ADOM",
            "Restormer",
            "Uformer",
            "NAFNet",
            "AnyIR",
            "TSFormer",
            "ShiftLUT",
            "MPMF-Net",
            "DCMGFormer",
            "JarvisIR",
            "GenDeg",
            "StripeRes-DominantRemove",
            "StripeRes-ResidualRemove",
            "StripeRes",
        ],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    summary = run_evaluation(
        config=read_json(config_path),
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir.resolve(),
        detector_name=args.detector,
        restorer_name=args.restorer,
        output_dir=args.output_dir.resolve(),
        limit=max(0, int(args.limit)),
        project_root=config_path.parent.parent,
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()

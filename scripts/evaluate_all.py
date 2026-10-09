from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from laserarmor.full_pipeline import merge_method_config, read_json, run_evaluation, write_json


def dataset_variants(root: Path) -> list[Path]:
    if (root / "manifest.jsonl").is_file():
        return [root]
    variants = sorted(path.parent for path in root.rglob("manifest.jsonl"))
    if not variants:
        raise RuntimeError(f"No manifest.jsonl files were found under {root}")
    return variants


def configured_datasets(config: dict[str, Any], root: Path) -> list[tuple[str, Path]]:
    datasets: list[tuple[str, Path]] = []
    for name, value in dict(config.get("datasets", {})).items():
        raw_path = value.get("path") if isinstance(value, dict) else value
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = root / path
        datasets.append((str(name), path.resolve()))
    if not datasets:
        raise RuntimeError("No datasets are configured")
    return datasets


def verify_assets(
    config: dict[str, Any],
    dataset_names: list[str],
    detectors: list[str],
    restorers: list[str],
    root: Path,
) -> None:
    missing: list[Path] = []
    for dataset_name in dataset_names:
        for category, methods in (("detectors", detectors), ("restorers", restorers)):
            for method in methods:
                entry = merge_method_config(config, category, method, dataset_name)
                for key in ("artifact", "checkpoint", "lut_path"):
                    if key not in entry:
                        continue
                    path = Path(str(entry[key]))
                    if not path.is_absolute():
                        path = root / path
                    if not path.exists():
                        missing.append(path.resolve())
    if missing:
        unique = sorted({str(path) for path in missing})
        raise RuntimeError("Missing required assets:\n" + "\n".join(unique))


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": summary["dataset"],
        "variant": summary["dataset_variant"],
        "detector": summary["detector"],
        "restorer": summary["restorer"],
        "sample_count": summary["sample_count"],
        **{f"detection_{key}": value for key, value in summary["detection"].items()},
        **{f"restoration_{key}": value for key, value in summary["restoration"].items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate every configured LaserArmor pipeline and dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/full_pipeline.json"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    root = config_path.parent.parent
    config = read_json(config_path)
    evaluation = dict(config.get("evaluation", {}))
    detectors = [str(value) for value in evaluation.get("detectors", [])]
    restorers = [str(value) for value in evaluation.get("restorers", [])]
    if not detectors or not restorers:
        raise RuntimeError("evaluation.detectors and evaluation.restorers must be non-empty")
    datasets = configured_datasets(config, root)
    verify_assets(config, [name for name, _ in datasets], detectors, restorers, root)
    runs: list[tuple[str, Path, str, str]] = []
    for dataset_name, dataset_root in datasets:
        for variant in dataset_variants(dataset_root):
            for detector in detectors:
                for restorer in restorers:
                    runs.append((dataset_name, variant, detector, restorer))
    if args.dry_run:
        print(json.dumps({"run_count": len(runs), "runs": [[name, str(variant), detector, restorer] for name, variant, detector, restorer in runs]}, indent=2, ensure_ascii=True))
        return
    maximum_images = evaluation.get("max_images", 0) if args.max_images is None else args.max_images
    output_root = args.output_root.resolve()
    summaries: list[dict[str, Any]] = []
    for dataset_name, variant, detector, restorer in runs:
        run_output = output_root / dataset_name / variant.name / detector / restorer
        summary = run_evaluation(
            config=config,
            dataset_name=dataset_name,
            dataset_dir=variant,
            detector_name=detector,
            restorer_name=restorer,
            output_dir=run_output,
            limit=max(0, int(maximum_images)),
            project_root=root,
        )
        summaries.append(summary)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(output_root / "all_metrics.json", summaries)
    flattened = [flatten_summary(summary) for summary in summaries]
    with (output_root / "all_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    print(json.dumps({"run_count": len(summaries), "metrics_csv": str(output_root / "all_metrics.csv"), "metrics_json": str(output_root / "all_metrics.json")}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()


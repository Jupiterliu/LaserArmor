from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from laserarmor.restoration.dominant_remove import StripeResDominantRemove


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Dataset must use NAME=PATH syntax")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Dataset name cannot be empty")
    return name, Path(path).expanduser().resolve()


def list_images(directory: Path) -> list[Path]:
    files = [path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort(key=lambda path: path.as_posix().lower())
    if not files:
        raise RuntimeError(f"No images were found under {directory}")
    return files


def resize_long_side(image: np.ndarray, target: int) -> np.ndarray:
    if target <= 0 or max(image.shape[:2]) == target:
        return image
    scale = float(target) / float(max(image.shape[:2]))
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def build_spans(height: int, style: str, generator: random.Random) -> list[tuple[int, int]]:
    if height < 2:
        return [(0, height)]
    if style == "single":
        stripe_height = min(height, max(1, round(height * generator.uniform(0.07, 0.18))))
        lower = min(height - stripe_height, max(0, round(height * 0.18)))
        upper = max(lower, min(height - stripe_height, round(height * 0.82) - stripe_height))
        start = generator.randint(lower, upper)
        return [(start, stripe_height)]
    if height < 6:
        split = max(1, height // 2)
        return [(0, 1), (split, 1)]
    first_height = min(height, max(1, round(height * generator.uniform(0.04, 0.09))))
    second_height = min(height, max(1, round(height * generator.uniform(0.04, 0.09))))
    first_lower = min(height - first_height, max(0, round(height * 0.15)))
    first_upper = max(first_lower, min(height - first_height, round(height * 0.48) - first_height))
    first = generator.randint(first_lower, first_upper)
    second_lower = min(height - second_height, max(first + first_height + 1, round(height * 0.52)))
    second_upper = max(second_lower, min(height - second_height, round(height * 0.88) - second_height))
    second = generator.randint(second_lower, second_upper)
    return [(first, first_height), (second, second_height)]


def synthesize_attack(
    clean_bgr: np.ndarray,
    color: str,
    style: str,
    generator: random.Random,
    noise_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    height, width = clean_bgr.shape[:2]
    spans = build_spans(height, style, generator)
    hard_mask = np.zeros((height, width), dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.float32)
    for start, stripe_height in spans:
        hard_mask[start : start + stripe_height] = 255
        alpha[start : start + stripe_height] = 1.0
    blur_sigma = max(1.0, height * 0.006)
    alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)
    x = np.linspace(0.0, 2.0 * np.pi, width, dtype=np.float32)
    horizontal = 0.78 + 0.22 * np.sin(x * generator.uniform(1.0, 4.0) + generator.uniform(0.0, 2.0 * np.pi))
    row_noise = noise_generator.normal(1.0, 0.08, size=(height, 1)).astype(np.float32)
    texture = np.clip(alpha * horizontal[None, :] * row_noise, 0.0, 1.0)
    amplitude = generator.uniform(120.0, 230.0)
    color_vector = np.asarray([0.06, 1.0, 0.06] if color == "green" else [0.06, 0.06, 1.0], dtype=np.float32)
    attacked = clean_bgr.astype(np.float32) + texture[:, :, None] * amplitude * color_vector[None, None, :]
    bloom = cv2.GaussianBlur(texture, (0, 0), max(2.0, blur_sigma * 2.0))
    attacked += bloom[:, :, None] * amplitude * 0.18
    return np.clip(np.round(attacked), 0, 255).astype(np.uint8), hard_mask, spans


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def prepare_variant(
    dataset_name: str,
    clean_root: Path,
    output_root: Path,
    color: str,
    style: str,
    seed: int,
    resize_to: int,
    maximum_images: int,
) -> int:
    files = list_images(clean_root)
    if maximum_images > 0:
        files = files[:maximum_images]
    variant_root = output_root / dataset_name / f"{color}-{style}"
    generator = random.Random(seed)
    noise_generator = np.random.default_rng(seed)
    dominant = StripeResDominantRemove()
    manifest_path = variant_root / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, source in enumerate(files):
            clean = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if clean is None:
                raise RuntimeError(f"Failed to read image: {source}")
            clean = resize_long_side(clean, resize_to)
            attacked, mask, spans = synthesize_attack(clean, color, style, generator, noise_generator)
            coarse, _ = dominant.restore(attacked, mask=mask, spans=spans)
            identifier = f"{index:06d}_{source.stem}"
            clean_rel = Path("clean") / f"{identifier}.png"
            attacked_rel = Path("attacked") / f"{identifier}.png"
            mask_rel = Path("mask") / f"{identifier}.png"
            coarse_rel = Path("coarse") / f"{identifier}.png"
            write_image(variant_root / clean_rel, clean)
            write_image(variant_root / attacked_rel, attacked)
            write_image(variant_root / mask_rel, mask)
            write_image(variant_root / coarse_rel, coarse)
            record = {
                "id": identifier,
                "source": source.name,
                "clean": clean_rel.as_posix(),
                "attacked": attacked_rel.as_posix(),
                "mask": mask_rel.as_posix(),
                "coarse": coarse_rel.as_posix(),
                "attack_color": color,
                "attack_style": style,
                "spans": spans,
            }
            manifest.write(json.dumps(record, ensure_ascii=True) + "\n")
    return len(files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paired LaserArmor evaluation data from clean images.")
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--colors", nargs="+", choices=["red", "green"], default=["red", "green"])
    parser.add_argument("--styles", nargs="+", choices=["single", "multi"], default=["single", "multi"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize-long-side", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    summary: dict[str, int] = {}
    for dataset_index, (dataset_name, clean_root) in enumerate(args.dataset):
        if not clean_root.is_dir():
            raise RuntimeError(f"Clean dataset directory was not found: {clean_root}")
        for color_index, color in enumerate(args.colors):
            for style_index, style in enumerate(args.styles):
                variant_seed = int(args.seed) + dataset_index * 10000 + color_index * 1000 + style_index * 100
                count = prepare_variant(
                    dataset_name,
                    clean_root,
                    output_root,
                    color,
                    style,
                    variant_seed,
                    int(args.resize_long_side),
                    max(0, int(args.max_images)),
                )
                summary[f"{dataset_name}/{color}-{style}"] = count
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

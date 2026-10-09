"""Create auditable, source-disjoint frame and physical-template manifests.

Input CSV columns:
  frames: dataset,image_path,source_sequence_id
  templates: template_path,source_template_id

Paths may be absolute or relative to their input CSV. A crop, channel swap, or
other derivative must retain its original source_template_id in the synthesis
record; this script deliberately does not infer provenance from file names.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {', '.join(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")
    for line, row in enumerate(rows, start=2):
        for field in required:
            if not row[field] or not row[field].strip():
                raise ValueError(f"{path}:{line} has an empty {field}")
    return rows


def resolve_source(csv_path: Path, value: str) -> Path:
    candidate = Path(value.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = csv_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def choose_frame_groups(
    groups: dict[tuple[str, str], list[dict[str, str]]], target: int, seed: int
) -> set[tuple[str, str]]:
    """Find an exact frame count without splitting any source sequence."""
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    reachable = 1  # Bit k means a subset of k frames is possible.
    limit = (1 << (target + 1)) - 1
    predecessor: dict[int, tuple[int, tuple[str, str]]] = {}
    for key in ordered:
        size = len(groups[key])
        newly_reachable = ((reachable << size) & limit) & ~reachable
        while newly_reachable:
            bit = newly_reachable & -newly_reachable
            total = bit.bit_length() - 1
            predecessor[total] = (total - size, key)
            newly_reachable ^= bit
        reachable |= (reachable << size) & limit
        if reachable & (1 << target):
            break
    if not reachable & (1 << target):
        nearest_below = reachable.bit_length() - 1
        raise ValueError(
            f"No source-sequence-disjoint split contains exactly {target} training "
            f"frames; the largest reachable count below it is {nearest_below}."
        )
    chosen: set[tuple[str, str]] = set()
    remaining = target
    while remaining:
        previous, key = predecessor[remaining]
        chosen.add(key)
        remaining = previous
    return chosen


def write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--templates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=35721,
                        help="Expected unique clean-frame count; use 0 to skip this check")
    parser.add_argument("--train-frames", type=int, default=3572)
    parser.add_argument("--train-template-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames_csv = args.frames_csv.resolve()
    templates_csv = args.templates_csv.resolve()
    frame_rows = read_rows(frames_csv, ("dataset", "image_path", "source_sequence_id"))
    template_rows = read_rows(templates_csv, ("template_path", "source_template_id"))
    if args.expected_total and len(frame_rows) != args.expected_total:
        raise ValueError(
            f"Expected {args.expected_total} clean frames but received {len(frame_rows)}; "
            "do not carry the manuscript counts over to a different data inventory."
        )
    if not 0 < args.train_frames < len(frame_rows):
        raise ValueError("--train-frames must be between 0 and the total frame count")
    if not 0 < args.train_template_fraction < 1:
        raise ValueError("--train-template-fraction must be strictly between 0 and 1")

    frame_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen_frames: set[Path] = set()
    for row in frame_rows:
        source = resolve_source(frames_csv, row["image_path"])
        if source in seen_frames:
            raise ValueError(f"Duplicate clean frame: {source}")
        seen_frames.add(source)
        row["image_path"] = str(source)
        row["dataset"] = row["dataset"].strip()
        row["source_sequence_id"] = row["source_sequence_id"].strip()
        frame_groups[(row["dataset"], row["source_sequence_id"])].append(row)
    train_frame_groups = choose_frame_groups(frame_groups, args.train_frames, args.seed)

    template_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_templates: set[Path] = set()
    for row in template_rows:
        source = resolve_source(templates_csv, row["template_path"])
        if source in seen_templates:
            raise ValueError(f"Duplicate template file: {source}")
        seen_templates.add(source)
        row["template_path"] = str(source)
        row["source_template_id"] = row["source_template_id"].strip()
        template_groups[row["source_template_id"]].append(row)
    if len(template_groups) < 2:
        raise ValueError("At least two independent source templates are required")
    ordered_templates = sorted(template_groups)
    random.Random(args.seed).shuffle(ordered_templates)
    train_template_count = max(
        1, min(len(ordered_templates) - 1,
               round(len(ordered_templates) * args.train_template_fraction))
    )
    train_template_groups = set(ordered_templates[:train_template_count])

    frame_output = []
    for key, group in sorted(frame_groups.items()):
        split = "train" if key in train_frame_groups else "test"
        frame_output.extend({"dataset": row["dataset"], "image_path": row["image_path"],
                             "source_sequence_id": row["source_sequence_id"], "split": split}
                            for row in group)
    template_output = []
    for key, group in sorted(template_groups.items()):
        split = "train" if key in train_template_groups else "test"
        template_output.extend({"template_path": row["template_path"],
                                "source_template_id": key, "split": split} for row in group)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "frames_split.csv", frame_output,
              ("dataset", "image_path", "source_sequence_id", "split"))
    write_csv(output_dir / "templates_split.csv", template_output,
              ("template_path", "source_template_id", "split"))
    summary = {
        "seed": args.seed,
        "train_frames": sum(row["split"] == "train" for row in frame_output),
        "test_frames": sum(row["split"] == "test" for row in frame_output),
        "train_sequences": len(train_frame_groups),
        "test_sequences": len(frame_groups) - len(train_frame_groups),
        "train_source_templates": len(train_template_groups),
        "test_source_templates": len(template_groups) - len(train_template_groups),
        "rule": "Every source sequence and source physical template belongs to exactly one split; all derivatives inherit their source split.",
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

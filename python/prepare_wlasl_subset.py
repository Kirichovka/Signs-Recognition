from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a smaller WLASL subset manifest for fast training runs."
    )
    parser.add_argument("--manifest", required=True, help="Input full JSONL manifest.")
    parser.add_argument("--output", required=True, help="Output subset JSONL manifest.")
    parser.add_argument(
        "--stats-output",
        default="",
        help="Optional output JSON path with subset statistics.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=50,
        help="Number of labels to keep, ranked by available sample count.",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=20,
        help="Minimum total samples a class must have to be eligible.",
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=80,
        help="Maximum train samples per class to keep.",
    )
    parser.add_argument(
        "--max-val-per-class",
        type=int,
        default=20,
        help="Maximum validation/test samples per class to keep.",
    )
    parser.add_argument(
        "--splits-for-validation",
        default="val,valid,validation,test",
        help="Comma-separated split names treated as validation.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_split(split: str, validation_splits: set[str]) -> str:
    split = (split or "train").strip().lower()
    if split in validation_splits:
        return "val"
    return "train"


def choose_labels(records: list[dict], num_classes: int, min_samples_per_class: int) -> list[str]:
    counts = Counter(record["label"] for record in records)
    eligible = [(label, count) for label, count in counts.items() if count >= min_samples_per_class]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return [label for label, _ in eligible[:num_classes]]


def build_subset(
    records: list[dict],
    selected_labels: set[str],
    max_train_per_class: int,
    max_val_per_class: int,
    validation_splits: set[str],
) -> tuple[list[dict], dict]:
    per_class_counter = defaultdict(lambda: {"train": 0, "val": 0})
    selected_records = []

    for record in records:
        label = record["label"]
        if label not in selected_labels:
            continue
        normalized_split = normalize_split(record.get("split", "train"), validation_splits)
        limit = max_val_per_class if normalized_split == "val" else max_train_per_class
        if per_class_counter[label][normalized_split] >= limit:
            continue
        updated = dict(record)
        updated["split"] = normalized_split
        selected_records.append(updated)
        per_class_counter[label][normalized_split] += 1

    stats = {
        "num_classes": len(selected_labels),
        "total_samples": len(selected_records),
        "classes": [
            {
                "label": label,
                "train_samples": per_class_counter[label]["train"],
                "val_samples": per_class_counter[label]["val"],
                "total_samples": per_class_counter[label]["train"] + per_class_counter[label]["val"],
            }
            for label in sorted(selected_labels)
        ],
    }
    return selected_records, stats


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    stats_output_path = Path(args.stats_output).resolve() if args.stats_output else output_path.with_suffix(".stats.json")
    validation_splits = {
        item.strip().lower()
        for item in args.splits_for_validation.split(",")
        if item.strip()
    }

    records = load_manifest(manifest_path)
    selected_labels = choose_labels(records, args.num_classes, args.min_samples_per_class)
    if len(selected_labels) < args.num_classes:
        raise ValueError(
            f"Only found {len(selected_labels)} eligible classes, fewer than requested {args.num_classes}."
        )

    subset_records, stats = build_subset(
        records=records,
        selected_labels=set(selected_labels),
        max_train_per_class=args.max_train_per_class,
        max_val_per_class=args.max_val_per_class,
        validation_splits=validation_splits,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in subset_records:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")

    stats_output_path.write_text(
        json.dumps(stats, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print(f"Selected {len(selected_labels)} classes")
    print(f"Wrote {len(subset_records)} subset samples to {output_path}")
    print(f"Wrote subset stats to {stats_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

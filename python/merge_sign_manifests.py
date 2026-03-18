from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a base sign manifest with an extra manifest, optionally remapping labels and forcing the extra split."
    )
    parser.add_argument("--base-manifest", required=True, help="Primary JSONL manifest, usually the dataset used for validation/test.")
    parser.add_argument("--extra-manifest", required=True, help="Additional JSONL manifest to merge into the base manifest.")
    parser.add_argument("--output", required=True, help="Output merged JSONL manifest.")
    parser.add_argument(
        "--label-map",
        default="",
        help="Optional CSV with columns source_label,target_label used to remap labels from the extra manifest.",
    )
    parser.add_argument(
        "--allowed-labels-file",
        default="",
        help="Optional text file with one allowed target label per line. If provided, only these labels are kept.",
    )
    parser.add_argument(
        "--force-extra-split",
        default="train",
        help="Split to assign to all extra-manifest records. Use an empty string to keep original splits.",
    )
    parser.add_argument(
        "--extra-source-name",
        default="extra",
        help="Value to store in the merged record under source_dataset for extra-manifest rows.",
    )
    parser.add_argument(
        "--base-source-name",
        default="base",
        help="Value to store in the merged record under source_dataset for base-manifest rows.",
    )
    parser.add_argument(
        "--stats-output",
        default="",
        help="Optional JSON output path for merge statistics.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_label_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"source_label", "target_label"}
        if not reader.fieldnames or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(f"Label map {path} must have CSV headers: source_label,target_label")
        for row in reader:
            source = (row.get("source_label") or "").strip()
            target = (row.get("target_label") or "").strip()
            if source and target:
                mapping[source] = target
    return mapping


def load_allowed_labels(path: Path) -> set[str]:
    labels: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            label = line.strip()
            if label and not label.startswith("#"):
                labels.add(label)
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def main() -> int:
    args = parse_args()
    base_manifest = Path(args.base_manifest).resolve()
    extra_manifest = Path(args.extra_manifest).resolve()
    output_path = Path(args.output).resolve()
    stats_output = Path(args.stats_output).resolve() if args.stats_output else output_path.with_suffix(".stats.json")

    base_records = load_jsonl(base_manifest)
    extra_records = load_jsonl(extra_manifest)
    label_map = load_label_map(Path(args.label_map).resolve()) if args.label_map else {}
    allowed_labels = load_allowed_labels(Path(args.allowed_labels_file).resolve()) if args.allowed_labels_file else None

    merged: list[dict] = []

    base_kept = 0
    for record in base_records:
        updated = dict(record)
        updated["source_dataset"] = args.base_source_name
        if allowed_labels and updated.get("label") not in allowed_labels:
            continue
        merged.append(updated)
        base_kept += 1

    extra_seen = 0
    extra_kept = 0
    extra_skipped_unmapped = 0
    extra_skipped_not_allowed = 0

    for record in extra_records:
        extra_seen += 1
        updated = dict(record)
        source_label = str(updated.get("label", "")).strip()
        if label_map:
            if source_label not in label_map:
                extra_skipped_unmapped += 1
                continue
            updated["label"] = label_map[source_label]
        if allowed_labels and updated.get("label") not in allowed_labels:
            extra_skipped_not_allowed += 1
            continue
        if args.force_extra_split:
            updated["split"] = args.force_extra_split
        updated["source_dataset"] = args.extra_source_name
        merged.append(updated)
        extra_kept += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in merged:
            stream.write(json.dumps(record, ensure_ascii=True) + "\n")

    stats = {
        "base_manifest": str(base_manifest),
        "extra_manifest": str(extra_manifest),
        "base_records_kept": base_kept,
        "extra_records_seen": extra_seen,
        "extra_records_kept": extra_kept,
        "extra_records_skipped_unmapped": extra_skipped_unmapped,
        "extra_records_skipped_not_allowed": extra_skipped_not_allowed,
        "merged_total": len(merged),
        "force_extra_split": args.force_extra_split,
        "base_source_name": args.base_source_name,
        "extra_source_name": args.extra_source_name,
        "allowed_labels_file": str(Path(args.allowed_labels_file).resolve()) if args.allowed_labels_file else "",
        "label_map": str(Path(args.label_map).resolve()) if args.label_map else "",
    }
    stats_output.write_text(json.dumps(stats, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Kept {base_kept} base records")
    print(f"Kept {extra_kept} extra records out of {extra_seen}")
    if label_map:
        print(f"Skipped {extra_skipped_unmapped} extra records because they were not mapped")
    if allowed_labels is not None:
        print(f"Skipped {extra_skipped_not_allowed} extra records because target labels were not allowed")
    print(f"Wrote merged manifest to {output_path}")
    print(f"Wrote merge stats to {stats_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

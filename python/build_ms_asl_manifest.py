from __future__ import annotations

import argparse
import json
from pathlib import Path

from sign_label_utils import build_target_label_index, load_labels_file, normalize_label_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a JSONL manifest from local MS-ASL clips."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the extracted MS-ASL annotation package root.",
    )
    parser.add_argument(
        "--clips-root",
        required=True,
        help="Path to the prepared local MS-ASL clips.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--labels-file",
        default="",
        help="Optional curated labels file. When provided, MS-ASL labels are mapped into this target label space.",
    )
    parser.add_argument(
        "--stats-output",
        default="",
        help="Optional output JSON path for manifest statistics.",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated MS-ASL splits to include.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_annotation_records(dataset_root: Path, splits: list[str]) -> list[tuple[str, int, dict]]:
    records: list[tuple[str, int, dict]] = []
    for split in splits:
        split_path = dataset_root / f"MSASL_{split}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing MS-ASL split file: {split_path}")
        data = load_json(split_path)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {split_path}, got {type(data).__name__}")
        for index, record in enumerate(data):
            records.append((split, index, record))
    return records


def build_synonym_target_map(synonym_groups: list[list[str]], target_index: dict[str, str]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for group in synonym_groups:
        normalized_group = [normalize_label_key(item) for item in group if normalize_label_key(item)]
        matches = [target_index[key] for key in normalized_group if key in target_index]
        if not matches:
            continue
        target_label = matches[0]
        for key in normalized_group:
            alias_map[key] = target_label
    return alias_map


def clip_name_for(split: str, index: int) -> str:
    return f"msasl_{split}_{index:06d}.mp4"


def resolve_target_label(record: dict, class_names: list[str], target_index: dict[str, str], alias_map: dict[str, str]) -> str:
    candidates = [
        record.get("clean_text"),
        record.get("text"),
        record.get("org_text"),
    ]
    class_id = record.get("label")
    if isinstance(class_id, int) and 0 <= class_id < len(class_names):
        candidates.append(class_names[class_id])

    for candidate in candidates:
        key = normalize_label_key(candidate)
        if not key:
            continue
        if key in target_index:
            return target_index[key]
        if key in alias_map:
            return alias_map[key]
    return ""


def normalize_split(split: str) -> str:
    split = split.strip().lower()
    if split in {"val", "valid", "validation"}:
        return "val"
    if split in {"test"}:
        return "test"
    return "train"


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    clips_root = Path(args.clips_root).resolve()
    output_path = Path(args.output).resolve()
    stats_path = Path(args.stats_output).resolve() if args.stats_output else output_path.with_suffix(".stats.json")
    splits = [item.strip().lower() for item in args.splits.split(",") if item.strip()]

    class_names = load_json(dataset_root / "MSASL_classes.json")
    synonym_groups = load_json(dataset_root / "MSASL_synonym.json")
    if not isinstance(class_names, list):
        raise ValueError("MSASL_classes.json must contain a list")
    if not isinstance(synonym_groups, list):
        raise ValueError("MSASL_synonym.json must contain a list")

    labels_file = Path(args.labels_file).resolve() if args.labels_file else None
    target_index: dict[str, str] = {}
    alias_map: dict[str, str] = {}
    if labels_file:
        labels = load_labels_file(labels_file)
        target_index = build_target_label_index(labels)
        alias_map = build_synonym_target_map(synonym_groups, target_index)

    written = 0
    missing_clips = 0
    skipped_unmapped = 0
    per_label_counts: dict[str, int] = {}
    per_split_counts: dict[str, int] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for split, index, record in load_annotation_records(dataset_root, splits):
            clip_path = clips_root / split / clip_name_for(split, index)
            if not clip_path.exists():
                missing_clips += 1
                continue

            target_label = ""
            if target_index:
                target_label = resolve_target_label(record, class_names, target_index, alias_map)
                if not target_label:
                    skipped_unmapped += 1
                    continue
            else:
                target_label = str(record.get("text") or record.get("clean_text") or "").strip()
                if not target_label:
                    skipped_unmapped += 1
                    continue

            normalized_split = normalize_split(split)
            item = {
                "dataset": "ms_asl",
                "label": target_label,
                "video_id": clip_path.stem,
                "video_path": str(clip_path),
                "split": normalized_split,
                "signer_id": record.get("signer_id"),
                "meta": record,
            }
            stream.write(json.dumps(item, ensure_ascii=True) + "\n")
            written += 1
            per_label_counts[target_label] = per_label_counts.get(target_label, 0) + 1
            per_split_counts[normalized_split] = per_split_counts.get(normalized_split, 0) + 1

    stats = {
        "dataset_root": str(dataset_root),
        "clips_root": str(clips_root),
        "labels_file": str(labels_file) if labels_file else "",
        "splits": splits,
        "written": written,
        "missing_clips": missing_clips,
        "skipped_unmapped": skipped_unmapped,
        "per_split_counts": per_split_counts,
        "per_label_counts": dict(sorted(per_label_counts.items())),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Wrote {written} MS-ASL samples to {output_path}")
    print(f"Missing local clips: {missing_clips}")
    print(f"Skipped unmapped samples: {skipped_unmapped}")
    print(f"Wrote stats to {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

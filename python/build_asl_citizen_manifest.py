from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a JSONL manifest from the ASL Citizen dataset."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the extracted ASL_Citizen folder.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL manifest path.",
    )
    return parser.parse_args()


def load_split_records(split_file: Path, split_name: str) -> list[dict]:
    records = []
    with split_file.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            normalized = {str(key).strip(): value for key, value in row.items()}
            normalized["split"] = split_name
            records.append(normalized)
    return records


def pick_video_column(record: dict) -> str:
    candidates = [
        "video_name",
        "video",
        "video_id",
        "instance_name",
        "id",
        "sample_name",
        "filename",
    ]
    for key in candidates:
        value = record.get(key)
        if value:
            return str(value).strip()
    raise KeyError(f"Could not find a video identifier column in record keys: {sorted(record.keys())}")


def pick_label_column(record: dict) -> str:
    candidates = [
        "gloss",
        "sign",
        "label",
        "lemma",
        "text",
    ]
    for key in candidates:
        value = record.get(key)
        if value:
            return str(value).strip()
    raise KeyError(f"Could not find a label column in record keys: {sorted(record.keys())}")


def find_video(videos_root: Path, stem: str) -> Path | None:
    stem = stem.strip()
    direct = videos_root / stem
    if direct.exists():
        return direct
    for extension in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        exact = videos_root / f"{stem}{extension}"
        if exact.exists():
            return exact
    matches = list(videos_root.rglob(f"{stem}.*"))
    return matches[0] if matches else None


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    splits_root = dataset_root / "splits"
    videos_root = dataset_root / "videos"
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train.csv": "train",
        "val.csv": "val",
        "valid.csv": "val",
        "validation.csv": "val",
        "test.csv": "test",
    }

    all_records = []
    discovered_split_files = []
    for file_name, split_name in split_map.items():
        split_file = splits_root / file_name
        if split_file.exists():
            discovered_split_files.append(split_file.name)
            all_records.extend(load_split_records(split_file, split_name))

    if not all_records:
        raise FileNotFoundError(f"No supported split CSV files were found in {splits_root}")

    written = 0
    missing = 0
    preview_columns = sorted(all_records[0].keys())

    with output_path.open("w", encoding="utf-8") as stream:
        for record in all_records:
            label = pick_label_column(record)
            video_stem = pick_video_column(record)
            video_path = find_video(videos_root, video_stem)
            if video_path is None:
                missing += 1
                continue
            item = {
                "dataset": "asl_citizen",
                "label": label,
                "video_id": video_path.stem,
                "video_path": str(video_path),
                "split": record["split"],
                "signer_id": record.get("signer_id") or record.get("participant_id") or record.get("signer"),
                "meta": record,
            }
            stream.write(json.dumps(item, ensure_ascii=True) + "\n")
            written += 1

    print(f"Discovered split files: {discovered_split_files}")
    print(f"Sample columns: {preview_columns}")
    print(f"Wrote {written} samples to {output_path}")
    print(f"Skipped {missing} rows because the matching video file was not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

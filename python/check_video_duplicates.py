from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a video dataset or manifest for exact duplicate files and possible split leakage."
    )
    parser.add_argument(
        "--root",
        default="",
        help="Optional dataset root folder to scan recursively for videos.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional JSONL manifest. If provided, paths and splits are read from the manifest.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path. Defaults to duplicate_report.json inside the scan root or next to the manifest.",
    )
    parser.add_argument(
        "--fail-on-cross-split",
        action="store_true",
        help="Exit with a non-zero code if duplicates are found across train/val/test-like splits.",
    )
    return parser.parse_args()


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_split(split: str) -> str:
    split = (split or "unknown").strip().lower()
    if split in {"training"}:
        return "train"
    if split in {"valid", "validation"}:
        return "val"
    return split or "unknown"


def infer_split_and_label_from_path(root: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(root)
    parts = list(relative.parts)
    split = "unknown"
    label = "unknown"
    split_candidates = {"train", "training", "val", "valid", "validation", "test"}
    for index, part in enumerate(parts):
        lower = part.lower()
        if lower in split_candidates:
            split = normalize_split(lower)
            if index + 1 < len(parts) - 1:
                label = parts[index + 1]
            break
    if label == "unknown" and len(parts) >= 2:
        label = parts[-2]
    return split, label


def load_targets_from_manifest(path: Path) -> tuple[list[dict], Path]:
    records: list[dict] = []
    root_hint: Path | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            video_path = Path(row["video_path"]).resolve()
            records.append(
                {
                    "path": video_path,
                    "relative_path": str(video_path),
                    "split": normalize_split(str(row.get("split", "unknown"))),
                    "label": str(row.get("label", "unknown")),
                }
            )
            if root_hint is None:
                root_hint = video_path.parent
    if not records:
        raise FileNotFoundError(f"No manifest rows were found in {path}")
    return records, (root_hint or path.parent)


def load_targets_from_root(root: Path) -> list[dict]:
    targets: list[dict] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            split, label = infer_split_and_label_from_path(root, path)
            targets.append(
                {
                    "path": path.resolve(),
                    "relative_path": str(path.resolve().relative_to(root.resolve())),
                    "split": split,
                    "label": label,
                }
            )
    return targets


def main() -> int:
    args = parse_args()
    if not args.root and not args.manifest:
        raise ValueError("Provide either --root or --manifest.")

    targets: list[dict]
    output_base: Path

    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        targets, output_base = load_targets_from_manifest(manifest_path)
        output = Path(args.output).resolve() if args.output else manifest_path.with_name(manifest_path.stem + "_duplicate_report.json")
    else:
        root = Path(args.root).resolve()
        targets = load_targets_from_root(root)
        output_base = root
        output = Path(args.output).resolve() if args.output else (root / "duplicate_report.json")

    if not targets:
        raise FileNotFoundError("No video files were found for duplicate scanning.")

    hash_groups: dict[str, list[dict]] = defaultdict(list)
    for item in targets:
        path = Path(item["path"])
        checksum = sha256sum(path)
        enriched = dict(item)
        enriched["sha256"] = checksum
        enriched["size_bytes"] = path.stat().st_size
        hash_groups[checksum].append(enriched)

    duplicate_groups = []
    cross_split_groups = []
    cross_label_groups = []

    for checksum, items in hash_groups.items():
        if len(items) < 2:
            continue
        splits = sorted({item["split"] for item in items})
        labels = sorted({item["label"] for item in items})
        group = {
            "sha256": checksum,
            "count": len(items),
            "splits": splits,
            "labels": labels,
            "items": items,
        }
        duplicate_groups.append(group)
        if len(splits) > 1:
            cross_split_groups.append(group)
        if len(labels) > 1:
            cross_label_groups.append(group)

    report = {
        "scan_base": str(output_base),
        "total_videos": len(targets),
        "duplicate_group_count": len(duplicate_groups),
        "cross_split_duplicate_groups": len(cross_split_groups),
        "cross_label_duplicate_groups": len(cross_label_groups),
        "duplicate_groups": duplicate_groups,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Scanned {len(targets)} video files")
    print(f"Found {len(duplicate_groups)} duplicate groups")
    print(f"Cross-split duplicate groups: {len(cross_split_groups)}")
    print(f"Cross-label duplicate groups: {len(cross_label_groups)}")
    print(f"Saved duplicate report to {output}")

    if args.fail_on_cross_split and cross_split_groups:
        print("Cross-split duplicate groups were found. Failing as requested.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

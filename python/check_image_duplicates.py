from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan an image dataset for exact duplicate files and possible split leakage."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder that contains the image dataset.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path. Defaults to <root>/duplicate_report.json",
    )
    parser.add_argument(
        "--fail-on-cross-split",
        action="store_true",
        help="Exit with a non-zero code if duplicates are found across train/test-like splits.",
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


def infer_split_and_label(root: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(root)
    parts = list(relative.parts)
    split = "unknown"
    label = "unknown"

    split_candidates = {"train", "training", "val", "valid", "validation", "test"}
    for index, part in enumerate(parts):
        lower = part.lower()
        if lower in split_candidates:
            split = lower
            if index + 1 < len(parts) - 1:
                label = parts[index + 1]
            break

    if label == "unknown" and len(parts) >= 2:
        label = parts[-2]
    return split, label


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve() if args.output else (root / "duplicate_report.json")

    image_paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise FileNotFoundError(f"No image files were found under {root}")

    hash_groups: dict[str, list[dict]] = defaultdict(list)
    for path in image_paths:
        checksum = sha256sum(path)
        split, label = infer_split_and_label(root, path)
        hash_groups[checksum].append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(root)),
                "split": split,
                "label": label,
                "size_bytes": path.stat().st_size,
            }
        )

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
        "root": str(root),
        "total_images": len(image_paths),
        "duplicate_group_count": len(duplicate_groups),
        "cross_split_duplicate_groups": len(cross_split_groups),
        "cross_label_duplicate_groups": len(cross_label_groups),
        "duplicate_groups": duplicate_groups,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Scanned {len(image_paths)} image files under {root}")
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

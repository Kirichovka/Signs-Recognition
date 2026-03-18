from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sign_label_utils import build_target_label_index, load_labels_file, normalize_label_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and locally clip an overlapping MS-ASL subset for augmentation."
    )
    parser.add_argument("--dataset-root", required=True, help="Path to the extracted MS-ASL annotation package.")
    parser.add_argument("--output-root", required=True, help="Target folder for clipped MS-ASL videos.")
    parser.add_argument(
        "--labels-file",
        required=True,
        help="Curated target label file used to decide which MS-ASL clips to keep.",
    )
    parser.add_argument(
        "--splits",
        default="train",
        help="Comma-separated MS-ASL splits to download. Defaults to train only for safe augmentation.",
    )
    parser.add_argument(
        "--max-clips-per-label",
        type=int,
        default=40,
        help="Maximum number of MS-ASL clips to keep per mapped label.",
    )
    parser.add_argument(
        "--report-output",
        default="",
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--cache-root",
        default="",
        help="Optional source-video cache root. Defaults to <output-root>/_source_cache.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild local clips even if they already exist.")
    return parser.parse_args()


def load_json(path: Path) -> list | dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def normalize_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return f"https://{text.lstrip('/')}"


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


def load_candidates(dataset_root: Path, labels_file: Path, splits: list[str]) -> list[dict[str, Any]]:
    class_names = load_json(dataset_root / "MSASL_classes.json")
    synonym_groups = load_json(dataset_root / "MSASL_synonym.json")
    target_index = build_target_label_index(load_labels_file(labels_file))
    alias_map = build_synonym_target_map(synonym_groups, target_index)

    counters: dict[str, int] = defaultdict(int)
    candidates: list[dict[str, Any]] = []
    for split in splits:
        split_path = dataset_root / f"MSASL_{split}.json"
        rows = load_json(split_path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected list in {split_path}")
        for index, record in enumerate(rows):
            target_label = resolve_target_label(record, class_names, target_index, alias_map)
            if not target_label:
                continue
            candidates.append(
                {
                    "split": split,
                    "index": index,
                    "target_label": target_label,
                    "record": record,
                    "clip_name": clip_name_for(split, index),
                }
            )
            counters[target_label] += 1
    return candidates


def download_source_video(url: str, cache_root: Path) -> Path:
    import yt_dlp

    cache_root.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "mp4/bestvideo+bestaudio/best",
        "outtmpl": str(cache_root / "%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "merge_output_format": "mp4",
        "restrictfilenames": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        download_path = Path(ydl.prepare_filename(info))
        if download_path.suffix.lower() != ".mp4":
            mp4_variant = download_path.with_suffix(".mp4")
            if mp4_variant.exists():
                download_path = mp4_variant
        if not download_path.exists():
            raise FileNotFoundError(f"yt-dlp reported {download_path}, but the file was not found")
        return download_path


def write_clip(source_path: Path, output_path: Path, record: dict) -> None:
    import cv2

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or float(record.get("fps") or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid source dimensions for {source_path}")

    start_time = float(record.get("start_time") or 0.0)
    end_time = float(record.get("end_time") or 0.0)
    start_frame = max(0, int(round(start_time * fps)))
    end_frame = max(start_frame + 1, int(round(end_time * fps)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open writer for {output_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    try:
        while frame_index < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Clip writer did not produce a valid file at {output_path}")


def main() -> int:
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # noqa: BLE001
        def tqdm(iterable, **_: object):  # type: ignore
            return iterable

    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    labels_file = Path(args.labels_file).resolve()
    cache_root = Path(args.cache_root).resolve() if args.cache_root else (output_root / "_source_cache")
    report_path = Path(args.report_output).resolve() if args.report_output else (output_root / "download_report.json")
    splits = [item.strip().lower() for item in args.splits.split(",") if item.strip()]

    candidates = load_candidates(dataset_root, labels_file, splits)
    selected: list[dict[str, Any]] = []
    per_label_kept: dict[str, int] = defaultdict(int)
    for item in candidates:
        label = item["target_label"]
        if per_label_kept[label] >= args.max_clips_per_label:
            continue
        selected.append(item)
        per_label_kept[label] += 1

    downloaded_sources: dict[str, str] = {}
    clips_created = 0
    clips_skipped_existing = 0
    failures: list[dict[str, str]] = []

    for item in tqdm(selected, desc="Preparing MS-ASL clips"):
        split = item["split"]
        clip_path = output_root / split / item["clip_name"]
        if clip_path.exists() and not args.force:
            clips_skipped_existing += 1
            continue

        url = normalize_url(item["record"].get("url"))
        if not url:
            failures.append({"clip": item["clip_name"], "reason": "missing_url"})
            continue

        try:
            if url not in downloaded_sources:
                downloaded_sources[url] = str(download_source_video(url, cache_root))
            source_path = Path(downloaded_sources[url])
            write_clip(source_path, clip_path, item["record"])
            clips_created += 1
        except Exception as error:  # noqa: BLE001
            failures.append({"clip": item["clip_name"], "reason": str(error)})

    report = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "labels_file": str(labels_file),
        "splits": splits,
        "selected_candidates": len(selected),
        "clips_created": clips_created,
        "clips_skipped_existing": clips_skipped_existing,
        "unique_source_urls_downloaded": len(downloaded_sources),
        "max_clips_per_label": args.max_clips_per_label,
        "per_label_selected": dict(sorted(per_label_kept.items())),
        "failures": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Selected {len(selected)} MS-ASL candidates")
    print(f"Created {clips_created} clips")
    print(f"Skipped {clips_skipped_existing} existing clips")
    print(f"Downloaded {len(downloaded_sources)} unique source videos")
    print(f"Wrote report to {report_path}")
    if failures:
        print(f"Failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

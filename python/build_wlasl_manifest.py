from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a JSONL manifest from the official WLASL annotation file."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to WLASL_v*.json from the official dataset release.",
    )
    parser.add_argument(
        "--videos-root",
        required=True,
        help="Root folder that contains downloaded WLASL video files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--extensions",
        default=".mp4,.mkv,.webm,.avi",
        help="Comma-separated list of video extensions to try.",
    )
    return parser.parse_args()


def find_video_file(videos_root: Path, video_id: str, extensions: list[str]) -> Path | None:
    for extension in extensions:
        exact = videos_root / f"{video_id}{extension}"
        if exact.exists():
            return exact
    for extension in extensions:
        matches = list(videos_root.rglob(f"{video_id}{extension}"))
        if matches:
            return matches[0]
    return None


def main() -> int:
    args = parse_args()
    annotations_path = Path(args.annotations).resolve()
    videos_root = Path(args.videos_root).resolve()
    output_path = Path(args.output).resolve()
    extensions = [item.strip() for item in args.extensions.split(",") if item.strip()]

    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0
    with output_path.open("w", encoding="utf-8") as stream:
        for gloss_entry in data:
            gloss = gloss_entry["gloss"]
            for instance in gloss_entry.get("instances", []):
                video_id = instance["video_id"]
                video_path = find_video_file(videos_root, video_id, extensions)
                if video_path is None:
                    missing += 1
                    continue
                record = {
                    "dataset": "wlasl",
                    "label": gloss,
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "split": instance.get("split", "train"),
                    "signer_id": instance.get("signer_id"),
                    "source": instance.get("source"),
                    "variation_id": instance.get("variation_id"),
                    "frame_start": instance.get("frame_start", 1),
                    "frame_end": instance.get("frame_end", -1),
                    "fps": instance.get("fps", 25),
                    "bbox": instance.get("bbox"),
                }
                stream.write(json.dumps(record, ensure_ascii=True) + "\n")
                written += 1

    print(f"Wrote {written} samples to {output_path}")
    print(f"Skipped {missing} samples because the video file was not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

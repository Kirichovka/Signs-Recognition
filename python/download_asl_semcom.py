from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = "https://zenodo.org/records/14635573/files/ASL_SemCom.zip?download=1"
DEFAULT_MD5 = "831ff816c3bb36ffc3b0c9f248cf5033"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the ASL_SemCom alphabet dataset from Zenodo."
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Target directory for the dataset. Defaults to <repo>/datasets/asl_semcom.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Dataset zip URL.",
    )
    parser.add_argument(
        "--expected-md5",
        default=DEFAULT_MD5,
        help="Expected MD5 checksum for the downloaded zip.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the zip even if it already exists.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download only and do not extract the archive.",
    )
    return parser.parse_args()


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            downloaded = block_count * block_size
            print(f"\rDownloaded {downloaded / (1024 * 1024):.1f} MB", end="", flush=True)
            return
        downloaded = min(block_count * block_size, total_size)
        percent = downloaded / total_size * 100
        print(
            f"\rDownloading {destination.name}: {percent:5.1f}% "
            f"({downloaded / (1024 * 1024):.1f} / {total_size / (1024 * 1024):.1f} MB)",
            end="",
            flush=True,
        )

    urllib.request.urlretrieve(url, destination, reporthook=report)
    print()


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (repo_root / "datasets" / "asl_semcom")
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "ASL_SemCom.zip"

    if zip_path.exists() and not args.force_download:
        print(f"Zip already exists: {zip_path}")
    else:
        print(f"Downloading dataset to {zip_path}")
        download_file(args.url, zip_path)

    checksum = md5sum(zip_path)
    print(f"MD5: {checksum}")
    if args.expected_md5 and checksum.lower() != args.expected_md5.lower():
        print(
            f"Checksum mismatch. Expected {args.expected_md5}, got {checksum}.",
            file=sys.stderr,
        )
        return 1

    if args.skip_extract:
        print("Skipping extraction by request.")
        return 0

    extracted_marker = output_dir / "ASL_SemCom"
    if extracted_marker.exists():
        print(f"Extraction target already exists: {extracted_marker}")
        return 0

    print(f"Extracting to {output_dir}")
    extract_zip(zip_path, output_dir)
    print("Extraction completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

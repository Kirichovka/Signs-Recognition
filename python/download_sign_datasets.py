from __future__ import annotations

import argparse
from pathlib import Path

from download_asl_semcom import DEFAULT_MD5 as ASL_SEMCOM_MD5
from download_asl_semcom import DEFAULT_URL as ASL_SEMCOM_URL
from download_asl_semcom import download_file, extract_zip, md5sum


ASL_CITIZEN_URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
MS_ASL_URL = "https://download.microsoft.com/download/3/c/a/3ca92c78-1c4a-4a91-a7ee-6980c1d242ec/MS-ASL.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the project datasets and prepare separate output folders for word and alphabet models."
    )
    parser.add_argument(
        "--datasets-root",
        default="",
        help="Target datasets root. Defaults to <repo>/datasets.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="",
        help="Target artifacts root. Defaults to <repo>/artifacts.",
    )
    parser.add_argument(
        "--skip-asl-citizen",
        action="store_true",
        help="Do not download ASL Citizen.",
    )
    parser.add_argument(
        "--skip-ms-asl",
        action="store_true",
        help="Do not download MS-ASL.",
    )
    parser.add_argument(
        "--skip-asl-semcom",
        action="store_true",
        help="Do not download ASL_SemCom.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download archives even if they already exist.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download archives without extracting them.",
    )
    return parser.parse_args()


def ensure_zip(
    url: str,
    zip_path: Path,
    *,
    expected_md5: str = "",
    force_download: bool = False,
) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists() and not force_download:
        print(f"Zip already exists: {zip_path}")
    else:
        print(f"Downloading {zip_path.name} -> {zip_path}")
        download_file(url, zip_path)

    if expected_md5:
        checksum = md5sum(zip_path)
        print(f"MD5 for {zip_path.name}: {checksum}")
        if checksum.lower() != expected_md5.lower():
            raise ValueError(f"Checksum mismatch for {zip_path.name}: expected {expected_md5}, got {checksum}")


def ensure_extract(zip_path: Path, extract_root: Path, extract_marker: Path) -> None:
    if extract_marker.exists():
        print(f"Already extracted: {extract_marker}")
        return
    print(f"Extracting {zip_path.name} -> {extract_root}")
    extract_zip(zip_path, extract_root)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    datasets_root = Path(args.datasets_root).resolve() if args.datasets_root else (repo_root / "datasets")
    artifacts_root = Path(args.artifacts_root).resolve() if args.artifacts_root else (repo_root / "artifacts")

    word_artifacts = artifacts_root / "word_model"
    alphabet_artifacts = artifacts_root / "alphabet_model"
    word_artifacts.mkdir(parents=True, exist_ok=True)
    alphabet_artifacts.mkdir(parents=True, exist_ok=True)
    print(f"Prepared word model artifacts dir: {word_artifacts}")
    print(f"Prepared alphabet model artifacts dir: {alphabet_artifacts}")

    if not args.skip_asl_citizen:
        citizen_root = datasets_root / "asl_citizen"
        citizen_zip = citizen_root / "ASL_Citizen.zip"
        ensure_zip(
            ASL_CITIZEN_URL,
            citizen_zip,
            force_download=args.force_download,
        )
        if not args.download_only:
            ensure_extract(citizen_zip, citizen_root, citizen_root / "ASL_Citizen")

    if not args.skip_ms_asl:
        ms_asl_root = datasets_root / "ms_asl"
        ms_asl_zip = ms_asl_root / "MS-ASL.zip"
        ensure_zip(
            MS_ASL_URL,
            ms_asl_zip,
            force_download=args.force_download,
        )
        if not args.download_only:
            # The package layout must still be inspected after extraction.
            ensure_extract(ms_asl_zip, ms_asl_root, ms_asl_root / "MS-ASL")

    if not args.skip_asl_semcom:
        semcom_root = datasets_root / "asl_semcom"
        semcom_zip = semcom_root / "ASL_SemCom.zip"
        ensure_zip(
            ASL_SEMCOM_URL,
            semcom_zip,
            expected_md5=ASL_SEMCOM_MD5,
            force_download=args.force_download,
        )
        if not args.download_only:
            ensure_extract(semcom_zip, semcom_root, semcom_root / "ASL_SemCom")

    print("Dataset bootstrap completed.")
    print("Recommended model layout:")
    print(f"  Word model outputs: {word_artifacts}")
    print(f"  Alphabet model outputs: {alphabet_artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

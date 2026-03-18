from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run word pipeline, alphabet pipeline, or both."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "word", "alphabet"),
        default="all",
        help="Which pipeline to run.",
    )
    parser.add_argument("--word-dataset-root", default="", help="ASL Citizen dataset root for the word pipeline.")
    parser.add_argument("--alphabet-dataset-root", default="", help="Alphabet image dataset root for the alphabet pipeline.")
    parser.add_argument("--word-run-name", default="everyday_daily_v1", help="Run name for the word pipeline.")
    parser.add_argument("--alphabet-run-name", default="alphabet_v1", help="Run name for the alphabet pipeline.")
    parser.add_argument("--word-extra-manifest", default="", help="Optional extra manifest for the word pipeline.")
    parser.add_argument("--word-extra-label-map", default="", help="Optional label map for the word pipeline.")
    parser.add_argument("--word-output-root", default="", help="Optional custom output root for the word pipeline.")
    parser.add_argument("--alphabet-output-root", default="", help="Optional custom output root for the alphabet pipeline.")
    parser.add_argument("--force", action="store_true", help="Forward force mode to child pipelines.")
    return parser.parse_args()


def run_step(command: list[str], cwd: Path) -> None:
    print()
    print(">>>", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python_dir = repo_root / "python"
    python_exe = sys.executable

    if args.mode in {"all", "word"}:
        if not args.word_dataset_root:
            raise ValueError("--word-dataset-root is required for mode=word or mode=all")
        word_command = [
            python_exe,
            str(python_dir / "run_word_model_pipeline.py"),
            "--dataset-root",
            str(Path(args.word_dataset_root).resolve()),
            "--run-name",
            args.word_run_name,
        ]
        if args.word_extra_manifest:
            word_command.extend(["--extra-manifest", str(Path(args.word_extra_manifest).resolve())])
        if args.word_extra_label_map:
            word_command.extend(["--extra-label-map", str(Path(args.word_extra_label_map).resolve())])
        if args.word_output_root:
            word_command.extend(["--output-root", str(Path(args.word_output_root).resolve())])
        if args.force:
            word_command.append("--force")
        run_step(word_command, cwd=repo_root)

    if args.mode in {"all", "alphabet"}:
        if not args.alphabet_dataset_root:
            raise ValueError("--alphabet-dataset-root is required for mode=alphabet or mode=all")
        alphabet_command = [
            python_exe,
            str(python_dir / "run_alphabet_model_pipeline.py"),
            "--dataset-root",
            str(Path(args.alphabet_dataset_root).resolve()),
            "--run-name",
            args.alphabet_run_name,
        ]
        if args.alphabet_output_root:
            alphabet_command.extend(["--output-root", str(Path(args.alphabet_output_root).resolve())])
        if args.force:
            alphabet_command.append("--force")
        run_step(alphabet_command, cwd=repo_root)

    print()
    print("Requested pipelines completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

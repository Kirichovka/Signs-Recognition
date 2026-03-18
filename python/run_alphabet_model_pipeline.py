from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end alphabet-model pipeline: duplicate scan, training, and optional ONNX export."
    )
    parser.add_argument("--dataset-root", required=True, help="Root folder containing alphabet train/test images.")
    parser.add_argument("--run-name", default="alphabet_v1", help="Name used for output files.")
    parser.add_argument("--output-root", default="", help="Output root. Defaults to <repo>/artifacts/alphabet_model/<run-name>.")
    parser.add_argument("--image-size", type=int, default=128, help="Square image size for training.")
    parser.add_argument("--epochs", type=int, default=16, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--skip-export", action="store_true", help="Skip ONNX export after training.")
    parser.add_argument("--skip-duplicate-check", action="store_true", help="Skip duplicate image scanning.")
    parser.add_argument("--allow-cross-split-duplicates", action="store_true", help="Do not fail if exact duplicate images are found across splits.")
    parser.add_argument("--force", action="store_true", help="Re-run intermediate steps even if outputs already exist.")
    return parser.parse_args()


def run_step(command: list[str], cwd: Path) -> None:
    print()
    print(">>>", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python_dir = repo_root / "python"
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else (repo_root / "artifacts" / "alphabet_model" / args.run_name)
    output_root.mkdir(parents=True, exist_ok=True)

    duplicate_report = output_root / f"{args.run_name}_duplicate_report.json"
    training_dir = output_root / "training_run"
    checkpoint_path = training_dir / "best_model.pt"
    onnx_path = output_root / f"{args.run_name}.onnx"
    metadata_path = output_root / f"{args.run_name}_metadata.json"
    summary_path = output_root / "pipeline_summary.json"
    python_exe = sys.executable

    if not args.skip_duplicate_check:
        duplicate_command = [
            python_exe,
            str(python_dir / "check_image_duplicates.py"),
            "--root",
            str(dataset_root),
            "--output",
            str(duplicate_report),
        ]
        if not args.allow_cross_split_duplicates:
            duplicate_command.append("--fail-on-cross-split")
        run_step(duplicate_command, cwd=repo_root)
    else:
        print("Skipping duplicate image check by request.")

    if args.force or not checkpoint_path.exists():
        run_step(
            [
                python_exe,
                str(python_dir / "train_alphabet_model.py"),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(training_dir),
                "--image-size",
                str(args.image_size),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--seed",
                str(args.seed),
            ],
            cwd=repo_root,
        )
    else:
        print(f"Skipping alphabet training, already exists: {checkpoint_path}")

    if not args.skip_export:
        if args.force or not onnx_path.exists() or not metadata_path.exists():
            run_step(
                [
                    python_exe,
                    str(python_dir / "export_alphabet_model_onnx.py"),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--output",
                    str(onnx_path),
                    "--metadata-output",
                    str(metadata_path),
                    "--top-k",
                    "5",
                ],
                cwd=repo_root,
            )
        else:
            print(f"Skipping alphabet ONNX export, already exists: {onnx_path}")

    summary = {
        "run_name": args.run_name,
        "dataset_root": str(dataset_root),
        "duplicate_report": "" if args.skip_duplicate_check else str(duplicate_report),
        "training_dir": str(training_dir),
        "checkpoint": str(checkpoint_path),
        "onnx": "" if args.skip_export else str(onnx_path),
        "metadata": "" if args.skip_export else str(metadata_path),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "skip_duplicate_check": args.skip_duplicate_check,
        "allow_cross_split_duplicates": args.allow_cross_split_duplicates,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print()
    print("Alphabet pipeline completed successfully.")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

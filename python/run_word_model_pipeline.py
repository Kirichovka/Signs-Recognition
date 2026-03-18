from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end word-model pipeline: build manifest, prepare subset, optionally merge extra data, extract features, train, and export ONNX."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the extracted ASL_Citizen dataset root.",
    )
    parser.add_argument(
        "--run-name",
        default="everyday_daily_v1",
        help="Name used for the output folder and generated files.",
    )
    parser.add_argument(
        "--labels-file",
        default="",
        help="Exact-label file for curated subset selection. Defaults to python/label_sets/asl_citizen_daily_v1.txt.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Output root for pipeline artifacts. Defaults to <repo>/artifacts/word_model/<run-name>.",
    )
    parser.add_argument(
        "--extra-manifest",
        default="",
        help="Optional second manifest, for example an MS-ASL overlap manifest.",
    )
    parser.add_argument(
        "--extra-label-map",
        default="",
        help="Optional CSV mapping extra-manifest labels into the base label space.",
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=120,
        help="Maximum train samples per class for the curated ASL Citizen subset.",
    )
    parser.add_argument(
        "--max-val-per-class",
        type=int,
        default=30,
        help="Maximum validation samples per class for the curated ASL Citizen subset.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=40,
        help="Frames per sequence for landmark extraction.",
    )
    parser.add_argument("--epochs", type=int, default=16, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--hidden-size", type=int, default=192, help="GRU hidden size.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout value.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export after training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run intermediate steps even if output files already exist.",
    )
    return parser.parse_args()


def run_step(command: list[str], cwd: Path) -> None:
    print()
    print(">>>", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python_dir = repo_root / "python"
    dataset_root = Path(args.dataset_root).resolve()
    labels_file = Path(args.labels_file).resolve() if args.labels_file else (python_dir / "label_sets" / "asl_citizen_daily_v1.txt")
    output_root = Path(args.output_root).resolve() if args.output_root else (repo_root / "artifacts" / "word_model" / args.run_name)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / f"{args.run_name}_asl_citizen_manifest.jsonl"
    subset_manifest_path = output_root / f"{args.run_name}_subset_manifest.jsonl"
    subset_stats_path = output_root / f"{args.run_name}_subset_stats.json"
    merged_manifest_path = output_root / f"{args.run_name}_merged_manifest.jsonl"
    merged_stats_path = output_root / f"{args.run_name}_merged_stats.json"
    features_path = output_root / f"{args.run_name}_features.npz"
    train_dir = output_root / "training_run"
    onnx_path = output_root / f"{args.run_name}.onnx"
    metadata_path = output_root / f"{args.run_name}_metadata.json"
    summary_path = output_root / "pipeline_summary.json"

    python_exe = sys.executable

    if args.force or not manifest_path.exists():
        ensure_parent(manifest_path)
        run_step(
            [
                python_exe,
                str(python_dir / "build_asl_citizen_manifest.py"),
                "--dataset-root",
                str(dataset_root),
                "--output",
                str(manifest_path),
            ],
            cwd=repo_root,
        )
    else:
        print(f"Skipping manifest build, already exists: {manifest_path}")

    if args.force or not subset_manifest_path.exists():
        ensure_parent(subset_manifest_path)
        run_step(
            [
                python_exe,
                str(python_dir / "prepare_wlasl_subset.py"),
                "--manifest",
                str(manifest_path),
                "--labels-file",
                str(labels_file),
                "--output",
                str(subset_manifest_path),
                "--stats-output",
                str(subset_stats_path),
                "--max-train-per-class",
                str(args.max_train_per_class),
                "--max-val-per-class",
                str(args.max_val_per_class),
            ],
            cwd=repo_root,
        )
    else:
        print(f"Skipping subset build, already exists: {subset_manifest_path}")

    final_manifest = subset_manifest_path
    final_manifest_stats = subset_stats_path
    if args.extra_manifest:
        extra_manifest = Path(args.extra_manifest).resolve()
        merge_command = [
            python_exe,
            str(python_dir / "merge_sign_manifests.py"),
            "--base-manifest",
            str(subset_manifest_path),
            "--extra-manifest",
            str(extra_manifest),
            "--allowed-labels-file",
            str(labels_file),
            "--force-extra-split",
            "train",
            "--extra-source-name",
            "ms_asl",
            "--base-source-name",
            "asl_citizen",
            "--output",
            str(merged_manifest_path),
            "--stats-output",
            str(merged_stats_path),
        ]
        if args.extra_label_map:
            merge_command.extend(["--label-map", str(Path(args.extra_label_map).resolve())])

        if args.force or not merged_manifest_path.exists():
            ensure_parent(merged_manifest_path)
            run_step(merge_command, cwd=repo_root)
        else:
            print(f"Skipping manifest merge, already exists: {merged_manifest_path}")
        final_manifest = merged_manifest_path
        final_manifest_stats = merged_stats_path

    if args.force or not features_path.exists():
        ensure_parent(features_path)
        run_step(
            [
                python_exe,
                str(python_dir / "extract_sign_features.py"),
                "--manifest",
                str(final_manifest),
                "--output",
                str(features_path),
                "--max-frames",
                str(args.max_frames),
            ],
            cwd=repo_root,
        )
    else:
        print(f"Skipping feature extraction, already exists: {features_path}")

    checkpoint_path = train_dir / "best_model.pt"
    if args.force or not checkpoint_path.exists():
        run_step(
            [
                python_exe,
                str(python_dir / "train_sign_model.py"),
                "--features",
                str(features_path),
                "--output-dir",
                str(train_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--hidden-size",
                str(args.hidden_size),
                "--dropout",
                str(args.dropout),
                "--lr",
                str(args.lr),
                "--seed",
                str(args.seed),
            ],
            cwd=repo_root,
        )
    else:
        print(f"Skipping training, already exists: {checkpoint_path}")

    if not args.skip_export:
        if args.force or not onnx_path.exists() or not metadata_path.exists():
            run_step(
                [
                    python_exe,
                    str(python_dir / "export_sign_model_onnx.py"),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--output",
                    str(onnx_path),
                    "--metadata-output",
                    str(metadata_path),
                    "--sequence-length",
                    str(args.max_frames),
                    "--top-k",
                    "5",
                ],
                cwd=repo_root,
            )
        else:
            print(f"Skipping ONNX export, already exists: {onnx_path}")

    summary = {
        "run_name": args.run_name,
        "dataset_root": str(dataset_root),
        "labels_file": str(labels_file),
        "base_manifest": str(manifest_path),
        "subset_manifest": str(subset_manifest_path),
        "final_manifest": str(final_manifest),
        "final_manifest_stats": str(final_manifest_stats),
        "features": str(features_path),
        "training_dir": str(train_dir),
        "checkpoint": str(checkpoint_path),
        "onnx": "" if args.skip_export else str(onnx_path),
        "metadata": "" if args.skip_export else str(metadata_path),
        "extra_manifest": str(Path(args.extra_manifest).resolve()) if args.extra_manifest else "",
        "extra_label_map": str(Path(args.extra_label_map).resolve()) if args.extra_label_map else "",
        "max_frames": args.max_frames,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print()
    print("Pipeline completed successfully.")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

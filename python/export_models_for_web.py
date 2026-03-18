from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export word model, alphabet model, or both into browser-ready ONNX artifacts."
    )
    parser.add_argument("--mode", choices=("all", "word", "alphabet"), default="all", help="Which model export path to run.")
    parser.add_argument("--word-checkpoint", default="", help="Path to the trained word-model checkpoint.")
    parser.add_argument("--alphabet-checkpoint", default="", help="Path to the trained alphabet-model checkpoint.")
    parser.add_argument("--word-name", default="word_model", help="Base name for the exported word-model web files.")
    parser.add_argument("--alphabet-name", default="alphabet_model", help="Base name for the exported alphabet-model web files.")
    parser.add_argument(
        "--output-root",
        default="",
        help="Root output folder for exported web artifacts. Defaults to <repo>/artifacts/web_exports.",
    )
    parser.add_argument(
        "--models-dir",
        default="",
        help="Optional models directory used when publish flags are enabled. Defaults to <repo>/models.",
    )
    parser.add_argument("--publish-word", action="store_true", help="Copy the exported word-model ONNX and metadata into the models directory.")
    parser.add_argument("--publish-alphabet", action="store_true", help="Copy the exported alphabet-model ONNX and metadata into the models directory.")
    parser.add_argument("--word-sequence-length", type=int, default=40, help="Sequence length metadata for the word model.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions the web UI should display.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version used for export.")
    return parser.parse_args()


def run_step(command: list[str], cwd: Path) -> None:
    print()
    print(">>>", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def copy_pair(source_onnx: Path, source_metadata: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_onnx, target_dir / source_onnx.name)
    shutil.copy2(source_metadata, target_dir / source_metadata.name)
    print(f"Published {source_onnx.name} -> {target_dir}")
    print(f"Published {source_metadata.name} -> {target_dir}")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python_dir = repo_root / "python"
    python_exe = sys.executable
    output_root = Path(args.output_root).resolve() if args.output_root else (repo_root / "artifacts" / "web_exports")
    models_dir = Path(args.models_dir).resolve() if args.models_dir else (repo_root / "models")

    if args.mode in {"all", "word"}:
        if not args.word_checkpoint:
            raise ValueError("--word-checkpoint is required for mode=word or mode=all")
        word_checkpoint = Path(args.word_checkpoint).resolve()
        word_out_dir = output_root / "word"
        word_onnx = word_out_dir / f"{args.word_name}.onnx"
        word_metadata = word_out_dir / f"{args.word_name}_metadata.json"
        run_step(
            [
                python_exe,
                str(python_dir / "export_sign_model_onnx.py"),
                "--checkpoint",
                str(word_checkpoint),
                "--output",
                str(word_onnx),
                "--metadata-output",
                str(word_metadata),
                "--sequence-length",
                str(args.word_sequence_length),
                "--top-k",
                str(args.top_k),
                "--opset",
                str(args.opset),
            ],
            cwd=repo_root,
        )
        if args.publish_word:
            copy_pair(word_onnx, word_metadata, models_dir)

    if args.mode in {"all", "alphabet"}:
        if not args.alphabet_checkpoint:
            raise ValueError("--alphabet-checkpoint is required for mode=alphabet or mode=all")
        alphabet_checkpoint = Path(args.alphabet_checkpoint).resolve()
        alphabet_out_dir = output_root / "alphabet"
        alphabet_onnx = alphabet_out_dir / f"{args.alphabet_name}.onnx"
        alphabet_metadata = alphabet_out_dir / f"{args.alphabet_name}_metadata.json"
        run_step(
            [
                python_exe,
                str(python_dir / "export_alphabet_model_onnx.py"),
                "--checkpoint",
                str(alphabet_checkpoint),
                "--output",
                str(alphabet_onnx),
                "--metadata-output",
                str(alphabet_metadata),
                "--top-k",
                str(args.top_k),
                "--opset",
                str(args.opset),
            ],
            cwd=repo_root,
        )
        if args.publish_alphabet:
            copy_pair(alphabet_onnx, alphabet_metadata, models_dir)

    print()
    print("Web export completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

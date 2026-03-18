from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_alphabet_model import AlphabetClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained alphabet image classifier to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--metadata-output", required=True, help="Output metadata JSON path")
    parser.add_argument("--top-k", type=int, default=5, help="How many predictions the browser UI should display")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    metadata_path = Path(args.metadata_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = AlphabetClassifier(
        num_classes=checkpoint["num_classes"],
        input_channels=checkpoint.get("input_channels", 3),
        image_size=checkpoint["image_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = checkpoint["image_size"]
    input_channels = checkpoint.get("input_channels", 3)
    dummy = torch.randn(1, input_channels, image_size, image_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    metadata = {
        "model_type": "image",
        "model_name": checkpoint_path.name,
        "onnx_name": output_path.name,
        "num_classes": checkpoint["num_classes"],
        "label_names": checkpoint["label_names"],
        "image_size": image_size,
        "input_channels": input_channels,
        "top_k": args.top_k,
        "input_name": "image",
        "output_name": "logits",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Saved ONNX model to {output_path}")
    print(f"Saved browser metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

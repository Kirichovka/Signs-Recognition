from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageBundle:
    images: np.ndarray
    labels: np.ndarray


class AlphabetDataset(Dataset):
    def __init__(self, bundle: ImageBundle) -> None:
        self.images = torch.tensor(bundle.images, dtype=torch.float32)
        self.labels = torch.tensor(bundle.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


class AlphabetClassifier(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 3, image_size: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.image_size = image_size
        self.input_channels = input_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a static ASL alphabet image classifier."
    )
    parser.add_argument("--dataset-root", required=True, help="Root directory containing train/test image folders.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and training metadata.")
    parser.add_argument("--image-size", type=int, default=128, help="Square image size used for training.")
    parser.add_argument("--epochs", type=int, default=16, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_split_images(dataset_root: Path, split_names: tuple[str, ...]) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    split_lookup = {name.lower() for name in split_names}
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = [part.lower() for part in path.relative_to(dataset_root).parts]
        split_index = next((idx for idx, part in enumerate(parts) if part in split_lookup), None)
        if split_index is None:
            continue
        relative_parts = path.relative_to(dataset_root).parts
        label = relative_parts[split_index + 1] if split_index + 1 < len(relative_parts) - 1 else path.parent.name
        candidates.append((path, label))
    return candidates


def load_image(path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return image


def build_bundle(samples: list[tuple[Path, str]], label_to_index: dict[str, int], image_size: int) -> ImageBundle:
    images = np.stack([load_image(path, image_size) for path, _ in samples], axis=0)
    labels = np.asarray([label_to_index[label] for _, label in samples], dtype=np.int64)
    return ImageBundle(images=images, labels=labels)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_inputs)
            losses.append(criterion(logits, batch_labels).item())
            accuracies.append(accuracy(logits, batch_labels))
    return float(np.mean(losses)), float(np.mean(accuracies))


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = collect_split_images(dataset_root, ("train", "training"))
    val_samples = collect_split_images(dataset_root, ("val", "valid", "validation", "test"))
    if not train_samples:
        raise FileNotFoundError(f"No train images found under {dataset_root}")
    if not val_samples:
        raise FileNotFoundError(f"No validation/test images found under {dataset_root}")

    label_names = sorted({label for _, label in train_samples + val_samples})
    label_to_index = {label: index for index, label in enumerate(label_names)}

    train_bundle = build_bundle(train_samples, label_to_index, args.image_size)
    val_bundle = build_bundle(val_samples, label_to_index, args.image_size)

    train_loader = DataLoader(AlphabetDataset(train_bundle), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(AlphabetDataset(val_bundle), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphabetClassifier(num_classes=len(label_names), image_size=args.image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_accuracy = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_accuracies = []
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accuracies.append(accuracy(logits.detach(), batch_labels))

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": float(np.mean(train_accuracies)),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(record)
        print(
            f"epoch={epoch} "
            f"train_loss={record['train_loss']:.4f} "
            f"train_acc={record['train_accuracy']:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(label_names),
                    "label_names": label_names,
                    "image_size": args.image_size,
                    "input_channels": 3,
                },
                output_dir / "best_model.pt",
            )

    metadata = {
        "dataset_root": str(dataset_root),
        "device": str(device),
        "num_classes": len(label_names),
        "label_names": label_names,
        "image_size": args.image_size,
        "best_val_accuracy": best_val_accuracy,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "history": history,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Saved checkpoint to {output_dir / 'best_model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class SplitBundle:
    sequences: np.ndarray
    labels: np.ndarray


class SequenceDataset(Dataset):
    def __init__(self, bundle: SplitBundle) -> None:
        self.sequences = torch.tensor(bundle.sequences, dtype=torch.float32)
        self.labels = torch.tensor(bundle.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


class SignSequenceClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.projection(inputs)
        encoded, _ = self.encoder(projected)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a temporal sign-language classifier from extracted landmark sequences."
    )
    parser.add_argument("--features", required=True, help="Path to NPZ features file.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metadata.")
    parser.add_argument("--epochs", type=int, default=18, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--hidden-size", type=int, default=192, help="GRU hidden size.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    sequences = data["sequences"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    splits = data["splits"].astype(str)
    label_names = [str(item) for item in data["label_names"].tolist()]
    return sequences, labels, splits, label_names


def make_splits(sequences: np.ndarray, labels: np.ndarray, splits: np.ndarray) -> tuple[SplitBundle, SplitBundle]:
    train_mask = np.isin(splits, ["train", "training"])
    val_mask = np.isin(splits, ["val", "valid", "validation", "test"])

    if not np.any(train_mask):
        raise ValueError("No training samples found in the features file.")
    if not np.any(val_mask):
        raise ValueError("No validation/test samples found in the features file.")

    train_bundle = SplitBundle(sequences=sequences[train_mask], labels=labels[train_mask])
    val_bundle = SplitBundle(sequences=sequences[val_mask], labels=labels[val_mask])
    return train_bundle, val_bundle


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
    features_path = Path(args.features).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences, labels, splits, label_names = load_features(features_path)
    train_bundle, val_bundle = make_splits(sequences, labels, splits)

    train_loader = DataLoader(SequenceDataset(train_bundle), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_bundle), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignSequenceClassifier(
        input_size=sequences.shape[-1],
        hidden_size=args.hidden_size,
        num_classes=len(label_names),
        dropout=args.dropout,
    ).to(device)

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
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_accuracy": float(np.mean(train_accuracies)),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch} "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"train_acc={epoch_record['train_accuracy']:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": sequences.shape[-1],
                    "hidden_size": args.hidden_size,
                    "num_classes": len(label_names),
                    "dropout": args.dropout,
                    "label_names": label_names,
                },
                output_dir / "best_model.pt",
            )

    metadata = {
        "features_path": str(features_path),
        "device": str(device),
        "num_classes": len(label_names),
        "sequence_length": int(sequences.shape[1]),
        "feature_size": int(sequences.shape[2]),
        "best_val_accuracy": best_val_accuracy,
        "label_names": label_names,
        "history": history,
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Saved checkpoint to {output_dir / 'best_model.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

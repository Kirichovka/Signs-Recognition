from __future__ import annotations

import re
from pathlib import Path


def normalize_label_key(label: str) -> str:
    text = str(label or "").strip().lower()
    text = re.sub(r"\d+$", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def load_labels_file(path: Path) -> list[str]:
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            label = line.strip()
            if label and not label.startswith("#"):
                labels.append(label)
    if not labels:
        raise ValueError(f"No labels were found in {path}.")
    return labels


def build_target_label_index(labels: list[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for label in labels:
        key = normalize_label_key(label)
        if not key:
            continue
        if key in index and index[key] != label:
            duplicates.setdefault(key, [index[key]])
            duplicates[key].append(label)
            continue
        index[key] = label
    if duplicates:
        detail = ", ".join(f"{key}: {sorted(set(values))}" for key, values in sorted(duplicates.items()))
        raise ValueError(f"Normalized label collisions detected: {detail}")
    return index

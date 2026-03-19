from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional at runtime
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(os.environ.get("SIGN_MODEL_PATH", REPO_ROOT.parents[1] / "AI-models" / "best_model.pt"))
DEFAULT_LANDMARKS_DATASET_PATH = Path(
    os.environ.get("LANDMARKS_DATASET_PATH", REPO_ROOT / "datasets" / "landmarks_dataset.json")
)
TOP_K = 5
K_NEIGHBORS = 5
Z_WEIGHT = 0.35


class PredictRequest(BaseModel):
    sequence: list[list[float]] = Field(..., description="Sequence of 40 frames with 154 features each")


class LandmarkPoint(BaseModel):
    id: int
    x: float
    y: float
    z: float


class LandmarkHand(BaseModel):
    handedness: str = "Unknown"
    score: float | None = None
    image_landmarks: list[LandmarkPoint] = Field(default_factory=list)


class LandmarkRecognizeRequest(BaseModel):
    mode: Literal["letters", "video"] = "letters"
    hands: list[LandmarkHand] = Field(default_factory=list)


class SequenceModelService:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.loaded = False
        self.label_names: list[str] = []
        self.sequence_length = 40
        self.feature_size = 0
        self.model = None
        self.load_error: str | None = None

        if torch is None:
            self.load_error = "torch is not installed in the current Python environment"
            return

        if not model_path.exists():
            self.load_error = f"model file was not found: {model_path}"
            return

        from train_sign_model import SignSequenceClassifier

        checkpoint = torch.load(model_path, map_location="cpu")
        self.label_names = checkpoint["label_names"]
        self.sequence_length = checkpoint.get("sequence_length", 40)
        self.feature_size = checkpoint["input_size"]
        self.model = SignSequenceClassifier(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_classes=checkpoint["num_classes"],
            dropout=checkpoint["dropout"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.loaded = True
        self.load_error = None

    def predict(self, sequence: list[list[float]]) -> dict:
        if not self.loaded or self.model is None:
            detail = self.load_error or f"sequence model is not loaded from {self.model_path}"
            raise HTTPException(status_code=503, detail=detail)
        if len(sequence) != self.sequence_length:
            raise HTTPException(status_code=400, detail=f"Expected {self.sequence_length} frames, got {len(sequence)}")
        for frame in sequence:
            if len(frame) != self.feature_size:
                raise HTTPException(status_code=400, detail=f"Expected frame width {self.feature_size}, got {len(frame)}")

        tensor = torch.tensor([sequence], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        top_scores, top_indices = torch.topk(probabilities, k=min(TOP_K, len(self.label_names)))
        predictions = [
            {"label": self.label_names[int(index)], "score": float(score)}
            for score, index in zip(top_scores.tolist(), top_indices.tolist())
        ]
        return {
            "predictions": predictions,
            "sequence_length": self.sequence_length,
            "feature_size": self.feature_size,
            "model_name": self.model_path.name,
        }

    def health(self) -> dict:
        return {
            "loaded": self.loaded,
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            "load_error": self.load_error,
            "num_classes": len(self.label_names),
            "label_names": self.label_names,
            "sequence_length": self.sequence_length,
            "feature_size": self.feature_size,
        }


class LandmarkDatasetService:
    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self.loaded = False
        self.sample_count = 0
        self.label_count = 0
        self.samples: list[dict] = []
        self.labels: list[str] = []
        self.modes: dict[str, dict] = {
            "letters": {"sample_count": 0, "label_count": 0, "labels": [], "samples": [], "label_counts": {}},
            "video": {"sample_count": 0, "label_count": 0, "labels": [], "samples": [], "label_counts": {}},
        }
        self.reload()

    @staticmethod
    def _is_letter_label(label: str) -> bool:
        return bool(re.fullmatch(r"[A-Z]", label.strip()))

    @staticmethod
    def _sort_landmarks(landmarks: list[dict]) -> list[dict] | None:
        sorted_landmarks = sorted(landmarks, key=lambda point: point["id"])
        return sorted_landmarks[:21] if len(sorted_landmarks) >= 21 else None

    @staticmethod
    def _normalize_landmarks(landmarks: list[dict]) -> list[dict] | None:
        if not landmarks or len(landmarks) < 21:
            return None

        wrist = landmarks[0]
        centered = [
            {
                "x": point["x"] - wrist["x"],
                "y": point["y"] - wrist["y"],
                "z": point["z"] - wrist["z"],
            }
            for point in landmarks
        ]
        scale = max(1e-6, *(math.hypot(point["x"], point["y"]) for point in centered))
        return [
            {
                "x": point["x"] / scale,
                "y": point["y"] / scale,
                "z": (point["z"] / scale) * Z_WEIGHT,
            }
            for point in centered
        ]

    @staticmethod
    def _flatten_landmarks(landmarks: list[dict]) -> list[float]:
        flattened: list[float] = []
        for point in landmarks:
            flattened.extend([point["x"], point["y"], point["z"]])
        return flattened

    @staticmethod
    def _mirror_vector(vector: list[float]) -> list[float]:
        mirrored = vector[:]
        for index in range(0, len(mirrored), 3):
            mirrored[index] *= -1
        return mirrored

    @staticmethod
    def _vector_distance(left_vector: list[float], right_vector: list[float]) -> float:
        total = 0.0
        point_count = len(left_vector) // 3
        for index in range(0, len(left_vector), 3):
            total += math.sqrt(
                (left_vector[index] - right_vector[index]) ** 2
                + (left_vector[index + 1] - right_vector[index + 1]) ** 2
                + (left_vector[index + 2] - right_vector[index + 2]) ** 2
            )
        return total / point_count

    def _build_mode_state(self, samples: list[dict], mode: Literal["letters", "video"]) -> dict:
        if mode == "letters":
            mode_samples = [sample for sample in samples if self._is_letter_label(sample["label"])]
        else:
            mode_samples = [sample for sample in samples if not self._is_letter_label(sample["label"])]

        label_counts: dict[str, int] = {}
        for sample in mode_samples:
            label_counts[sample["label"]] = label_counts.get(sample["label"], 0) + 1

        return {
            "sample_count": len(mode_samples),
            "label_count": len(label_counts),
            "labels": sorted(label_counts.keys()),
            "samples": mode_samples,
            "label_counts": label_counts,
        }

    def reload(self) -> dict:
        if not self.dataset_path.exists():
            self.loaded = False
            self.sample_count = 0
            self.label_count = 0
            self.samples = []
            self.labels = []
            self.modes = {
                "letters": {"sample_count": 0, "label_count": 0, "labels": [], "samples": [], "label_counts": {}},
                "video": {"sample_count": 0, "label_count": 0, "labels": [], "samples": [], "label_counts": {}},
            }
            return self.health()

        raw_dataset = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        normalized_samples: list[dict] = []
        label_counts: dict[str, int] = {}

        for sample in raw_dataset.get("samples", []):
            sample_hands = [
                hand for hand in sample.get("hands", [])
                if isinstance(hand.get("image_landmarks"), list) and len(hand["image_landmarks"]) >= 21
            ]
            if not sample_hands:
                continue

            primary_hand = max(sample_hands, key=lambda hand: float(hand.get("score") or 0.0))
            sorted_landmarks = self._sort_landmarks(primary_hand.get("image_landmarks", []))
            normalized = self._normalize_landmarks(sorted_landmarks or [])
            if not normalized:
                continue

            label = str(sample.get("label") or "unknown").strip()
            normalized_samples.append(
                {
                    "id": sample.get("id"),
                    "label": label,
                    "handedness": primary_hand.get("handedness") or "Unknown",
                    "handedness_score": float(primary_hand.get("score") or 0.0),
                    "vector": self._flatten_landmarks(normalized),
                }
            )
            label_counts[label] = label_counts.get(label, 0) + 1

        self.loaded = True
        self.samples = normalized_samples
        self.sample_count = len(normalized_samples)
        self.label_count = len(label_counts)
        self.labels = sorted(label_counts.keys())
        self.modes = {
            "letters": self._build_mode_state(normalized_samples, "letters"),
            "video": self._build_mode_state(normalized_samples, "video"),
        }
        return self.health()

    def labels_for_mode(self, mode: Literal["letters", "video"]) -> dict:
        mode_state = self.modes[mode]
        return {
            "mode": mode,
            "label_count": mode_state["label_count"],
            "sample_count": mode_state["sample_count"],
            "labels": mode_state["labels"],
            "label_counts": mode_state["label_counts"],
        }

    def recognize(self, payload: LandmarkRecognizeRequest) -> dict:
        if not self.loaded or not self.sample_count:
            raise HTTPException(status_code=503, detail=f"Landmark dataset is not loaded from {self.dataset_path}")
        if not payload.hands:
            raise HTTPException(status_code=400, detail="No hands were provided.")

        mode_state = self.modes[payload.mode]
        if not mode_state["sample_count"]:
            raise HTTPException(status_code=400, detail=f"No usable samples are available in {payload.mode} mode.")

        usable_hands = [hand for hand in payload.hands if len(hand.image_landmarks) >= 21]
        if not usable_hands:
            raise HTTPException(status_code=400, detail="No hand contained at least 21 landmarks.")

        primary_hand = max(usable_hands, key=lambda hand: float(hand.score or 0.0))
        sorted_landmarks = self._sort_landmarks([point.model_dump() for point in primary_hand.image_landmarks])
        normalized_query = self._normalize_landmarks(sorted_landmarks or [])
        if not normalized_query:
            raise HTTPException(status_code=400, detail="Could not normalize the provided hand landmarks.")

        query_vector = self._flatten_landmarks(normalized_query)
        mirrored_query = self._mirror_vector(query_vector)
        distances = []

        for sample in mode_state["samples"]:
            distance = min(
                self._vector_distance(query_vector, sample["vector"]),
                self._vector_distance(mirrored_query, sample["vector"]),
            )
            distances.append(
                {
                    "label": sample["label"],
                    "distance": distance,
                    "sample_id": sample["id"],
                }
            )

        if not distances:
            raise HTTPException(status_code=400, detail=f"No normalized samples are available in {payload.mode} mode.")

        distances.sort(key=lambda item: item["distance"])
        neighbors = distances[: min(K_NEIGHBORS, len(distances))]

        label_scores: dict[str, float] = {}
        label_best_distances: dict[str, float] = {}
        for neighbor in neighbors:
            vote = math.exp(-4.0 * neighbor["distance"])
            label = neighbor["label"]
            label_scores[label] = label_scores.get(label, 0.0) + vote
            if label not in label_best_distances or neighbor["distance"] < label_best_distances[label]:
                label_best_distances[label] = neighbor["distance"]

        ranked_labels = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        predicted_label, predicted_weight = ranked_labels[0]
        total_weight = sum(label_scores.values())
        best_distance = label_best_distances[predicted_label]
        confidence = predicted_weight / total_weight if total_weight else 0.0
        similarity = 1.0 / (1.0 + best_distance)

        top_matches = [
            {
                "label": label,
                "vote": score,
                "best_distance": label_best_distances[label],
            }
            for label, score in ranked_labels[:3]
        ]

        return {
            "mode": payload.mode,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "similarity": similarity,
            "best_distance": best_distance,
            "top_matches": top_matches,
            "neighbors": neighbors,
            "live_handedness": primary_hand.handedness,
            "live_handedness_score": float(primary_hand.score or 0.0),
            "mode_sample_count": mode_state["sample_count"],
            "mode_label_count": mode_state["label_count"],
        }

    def health(self) -> dict:
        return {
            "loaded": self.loaded,
            "dataset_path": str(self.dataset_path),
            "sample_count": self.sample_count,
            "label_count": self.label_count,
            "labels": self.labels,
            "modes": {
                key: {
                    "sample_count": value["sample_count"],
                    "label_count": value["label_count"],
                    "labels": value["labels"],
                    "label_counts": value["label_counts"],
                }
                for key, value in self.modes.items()
            },
        }


app = FastAPI(title="Local ASL Recognition API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sequence_model_service = SequenceModelService(DEFAULT_MODEL_PATH)
landmark_dataset_service = LandmarkDatasetService(DEFAULT_LANDMARKS_DATASET_PATH)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "sequence_model": sequence_model_service.health(),
        "landmark_dataset": landmark_dataset_service.health(),
        "capabilities": {
            "sequence_prediction": sequence_model_service.loaded,
            "landmark_knn_recognition": landmark_dataset_service.loaded and landmark_dataset_service.sample_count > 0,
        },
    }


@app.get("/api/landmarks/stats")
def landmark_stats() -> dict:
    return landmark_dataset_service.health()


@app.get("/api/landmarks/labels")
def landmark_labels(mode: Literal["letters", "video"] | None = Query(default=None)) -> dict:
    if mode:
        return landmark_dataset_service.labels_for_mode(mode)
    return {
        "all": {
            "label_count": landmark_dataset_service.label_count,
            "sample_count": landmark_dataset_service.sample_count,
            "labels": landmark_dataset_service.labels,
        },
        "letters": landmark_dataset_service.labels_for_mode("letters"),
        "video": landmark_dataset_service.labels_for_mode("video"),
    }


@app.post("/api/landmarks/reload")
def reload_landmark_dataset() -> dict:
    return landmark_dataset_service.reload()


@app.post("/api/recognize/landmarks")
def recognize_landmarks(payload: LandmarkRecognizeRequest) -> dict:
    return landmark_dataset_service.recognize(payload)


@app.post("/api/predict")
def predict_sequence(payload: PredictRequest) -> dict:
    return sequence_model_service.predict(payload.sequence)


@app.post("/api/recognize/sequence")
def recognize_sequence(payload: PredictRequest) -> dict:
    return sequence_model_service.predict(payload.sequence)


app.mount("/", StaticFiles(directory=REPO_ROOT, html=True), name="static")

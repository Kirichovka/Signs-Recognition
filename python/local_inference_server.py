from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from train_sign_model import SignSequenceClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(os.environ.get("SIGN_MODEL_PATH", REPO_ROOT.parents[1] / "AI-models" / "best_model.pt"))
TOP_K = 5


class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(..., description="Sequence of 40 frames with 154 features each")


class ModelService:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file was not found: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model_path = model_path
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

    def predict(self, sequence: List[List[float]]) -> dict:
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
        }


app = FastAPI(title="Local ASL Inference Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService(DEFAULT_MODEL_PATH)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_name": model_service.model_path.name,
        "num_classes": len(model_service.label_names),
        "label_names": model_service.label_names,
        "sequence_length": model_service.sequence_length,
        "feature_size": model_service.feature_size,
    }


@app.post("/api/predict")
def predict(payload: PredictRequest) -> dict:
    return model_service.predict(payload.sequence)


app.mount("/", StaticFiles(directory=REPO_ROOT, html=True), name="static")

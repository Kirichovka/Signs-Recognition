# Gesture Trainer

Standalone project for simple child-friendly hand sign practice.

## Web version

```powershell
cd D:\Integration-Game\gesture-trainer-web
python -m http.server 4174
```

Open:

- `http://127.0.0.1:4174`

Notes:

- The web app uses MediaPipe Hands from CDN.
- Camera access must be allowed in the browser.
- If the camera is busy, close other apps and press `Retry Camera`.

## Python version

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python gesture_trainer.py
```

Controls:

- `N` - next gesture
- `P` - previous gesture
- `R` - reset hold timer
- `Q` or `Esc` - quit

## Training a real sign-language model

The current webcam demo is still a prototype. For real sign-language recognition, the next step is to train on a real isolated-sign video dataset and switch from hand-written rules to a temporal model.

Recommended starting point:

- `WLASL` for word-level American Sign Language research and benchmarking.
- `ASL Citizen` if you want a dataset closer to webcam dictionary-style retrieval.

This repo now includes a small training pipeline for real video data:

1. Build a manifest from downloaded WLASL videos:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python build_wlasl_manifest.py `
  --annotations D:\datasets\wlasl\WLASL_v0.3.json `
  --videos-root D:\datasets\wlasl\videos `
  --output D:\datasets\wlasl\wlasl_manifest.jsonl
```

2. Extract temporal landmark features from the videos:

```powershell
python extract_sign_features.py `
  --manifest D:\datasets\wlasl\wlasl_manifest.jsonl `
  --output D:\datasets\wlasl\wlasl_features.npz `
  --max-frames 48
```

3. Train a temporal classifier on the extracted sequences:

```powershell
python train_sign_model.py `
  --features D:\datasets\wlasl\wlasl_features.npz `
  --output-dir D:\datasets\wlasl\training_run `
  --epochs 18 `
  --batch-size 16
```

Outputs:

- `best_model.pt` - trained PyTorch checkpoint
- `training_metrics.json` - training and validation metrics

Notes:

- This pipeline is for isolated sign recognition, not full sentence-level translation.
- `extract_sign_features.py` uses MediaPipe Holistic to encode both hands and upper-body pose over time.
- The model is a GRU-based temporal classifier over landmark sequences, which is a much better baseline for real sign videos than per-frame finger rules.

## Fast WLASL-50 setup for NVIDIA Brev

If you want to stay inside a 5-6 hour session, use a 50-class subset first.

Recommended flow:

1. Build the full manifest:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python build_wlasl_manifest.py `
  --annotations /workspace/datasets/wlasl/WLASL_v0.3.json `
  --videos-root /workspace/datasets/wlasl/videos `
  --output /workspace/datasets/wlasl/wlasl_manifest.jsonl
```

2. Prepare a 50-class subset with capped samples per class:

```powershell
python prepare_wlasl_subset.py `
  --manifest /workspace/datasets/wlasl/wlasl_manifest.jsonl `
  --output /workspace/datasets/wlasl/wlasl_50_manifest.jsonl `
  --stats-output /workspace/datasets/wlasl/wlasl_50_stats.json `
  --num-classes 50 `
  --min-samples-per-class 20 `
  --max-train-per-class 80 `
  --max-val-per-class 20
```

3. Extract features from the 50-class subset:

```powershell
python extract_sign_features.py `
  --manifest /workspace/datasets/wlasl/wlasl_50_manifest.jsonl `
  --output /workspace/datasets/wlasl/wlasl_50_features.npz `
  --max-frames 40
```

4. Train the baseline model:

```powershell
python train_sign_model.py `
  --features /workspace/datasets/wlasl/wlasl_50_features.npz `
  --output-dir /workspace/datasets/wlasl/wlasl_50_run `
  --epochs 16 `
  --batch-size 32 `
  --hidden-size 192
```

Suggested Brev session target:

- `50` classes
- `40` frames per video
- up to `80` train videos per class
- up to `20` validation videos per class

This is the best balance in this repo right now between realism, speed, and GPU cost.

## Alternative: ASL Citizen quick start

If WLASL external links are too unreliable, switch to `ASL Citizen`. It is distributed as one official archive, which is much more practical for a time-limited GPU session.

1. Build a manifest from the extracted ASL Citizen dataset:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python build_asl_citizen_manifest.py `
  --dataset-root /workspace/datasets/asl_citizen/ASL_Citizen `
  --output /workspace/datasets/asl_citizen/asl_citizen_manifest.jsonl
```

2. Prepare a 50-class subset:

```powershell
python prepare_wlasl_subset.py `
  --manifest /workspace/datasets/asl_citizen/asl_citizen_manifest.jsonl `
  --output /workspace/datasets/asl_citizen/asl_citizen_50_manifest.jsonl `
  --stats-output /workspace/datasets/asl_citizen/asl_citizen_50_stats.json `
  --num-classes 50 `
  --min-samples-per-class 20 `
  --max-train-per-class 80 `
  --max-val-per-class 20
```

3. Extract features and train exactly the same way as for WLASL:

```powershell
python extract_sign_features.py `
  --manifest /workspace/datasets/asl_citizen/asl_citizen_50_manifest.jsonl `
  --output /workspace/datasets/asl_citizen/asl_citizen_50_features.npz `
  --max-frames 40

python train_sign_model.py `
  --features /workspace/datasets/asl_citizen/asl_citizen_50_features.npz `
  --output-dir /workspace/datasets/asl_citizen/asl_citizen_50_run `
  --epochs 16 `
  --batch-size 32 `
  --hidden-size 192
```

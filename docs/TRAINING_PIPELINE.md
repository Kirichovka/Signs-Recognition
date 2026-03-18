# Training Pipeline

## 1. Goal

This document describes the full workflow for training a new model:

1. download the dataset
2. build a manifest
3. select a subset
4. extract landmark features
5. train a PyTorch model
6. export the model to ONNX
7. replace the browser model files in the project

The main recommended dataset right now is:

- `ASL Citizen`

Why:

- one official archive
- fewer reliability problems than WLASL external links
- well suited to isolated sign recognition

## 2. Requirements

### 2.1 Local environment

- Python `3.10+`
- `pip`
- `venv`

### 2.2 Linux GPU server

Reasonable minimum:

- NVIDIA GPU
- `4+ CPU`
- `16+ GB RAM`
- enough disk space for the archive, extracted videos, features, and checkpoints

For OpenCV on Ubuntu, you may also need:

```bash
sudo apt update
sudo apt install -y unzip libgl1 libglib2.0-0
```

## 3. Install Python dependencies

```bash
cd ~/workspace/Signs-Recognition/python
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Download ASL Citizen

```bash
mkdir -p ~/workspace/datasets/asl_citizen
cd ~/workspace/datasets/asl_citizen
wget https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip
unzip -q ASL_Citizen.zip
```

Expected structure:

```text
~/workspace/datasets/asl_citizen/
  ASL_Citizen.zip
  ASL_Citizen/
    splits/
    videos/
```

## 5. Build the manifest

```bash
cd ~/workspace/Signs-Recognition/python
source .venv/bin/activate

python build_asl_citizen_manifest.py \
  --dataset-root ~/workspace/datasets/asl_citizen/ASL_Citizen \
  --output ~/workspace/datasets/asl_citizen/asl_citizen_manifest.jsonl
```

This step:

- reads the official split CSV files
- resolves video paths
- writes a JSONL manifest

## 6. Create a subset

### 6.1 Automatic top-class selection

```bash
python prepare_wlasl_subset.py \
  --manifest ~/workspace/datasets/asl_citizen/asl_citizen_manifest.jsonl \
  --output ~/workspace/datasets/asl_citizen/asl_citizen_50_manifest.jsonl \
  --stats-output ~/workspace/datasets/asl_citizen/asl_citizen_50_stats.json \
  --num-classes 50 \
  --min-samples-per-class 20 \
  --max-train-per-class 80 \
  --max-val-per-class 20
```

### 6.2 Curated subset from an explicit label list

For practical everyday signs:

```bash
python prepare_wlasl_subset.py \
  --manifest ~/workspace/datasets/asl_citizen/asl_citizen_manifest.jsonl \
  --labels-file ~/workspace/Signs-Recognition/python/label_sets/asl_citizen_daily_v1.txt \
  --output ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_manifest.jsonl \
  --stats-output ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_stats.json \
  --max-train-per-class 120 \
  --max-val-per-class 30
```

Important note:

- if the dataset only contains about `30` samples for a sign, larger caps will not produce more data
- the limits are upper bounds, not guaranteed class sizes

## 7. Inspect subset size

Example:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_stats.json').expanduser()
data = json.loads(path.read_text(encoding='utf-8'))

print('num_classes =', data['num_classes'])
print('total_samples =', data['total_samples'])
for item in data['classes']:
    print(item['label'], item['train_samples'], item['val_samples'], item['total_samples'])
PY
```

## 8. Extract features

```bash
python extract_sign_features.py \
  --manifest ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_manifest.jsonl \
  --output ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_features.npz \
  --max-frames 40
```

What extraction does:

- opens each video
- runs frames through MediaPipe Holistic
- collects:
  - left hand
  - right hand
  - upper pose landmarks
- normalizes the coordinates
- builds a fixed-length sequence
- saves everything into an `.npz`

Result contents:

- `sequences`
- `labels`
- `splits`
- `label_names`
- `feature_size`

## 9. Check the feature file

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

p = Path('~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_features.npz').expanduser()
data = np.load(p, allow_pickle=True)
print('sequences shape =', data['sequences'].shape)
print('labels shape =', data['labels'].shape)
print('num_classes =', len(data['label_names']))
print('feature_size =', int(data['feature_size'][0]))
PY
```

## 10. Train the model

Example baseline:

```bash
python train_sign_model.py \
  --features ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_features.npz \
  --output-dir ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_run \
  --epochs 16 \
  --batch-size 32 \
  --hidden-size 192
```

Expected outputs:

- `best_model.pt`
- `training_metrics.json`

## 11. Check the metrics

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path('~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_run/training_metrics.json').expanduser()
data = json.loads(p.read_text(encoding='utf-8'))
print('best_val_accuracy =', data['best_val_accuracy'])
print('last_epoch =', data['history'][-1])
PY
```

## 12. Export to ONNX

```bash
python export_sign_model_onnx.py \
  --checkpoint ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_run/best_model.pt \
  --output ~/workspace/Signs-Recognition/models/asl_citizen_daily_v1.onnx \
  --metadata-output ~/workspace/Signs-Recognition/models/asl_citizen_daily_v1_metadata.json \
  --sequence-length 40 \
  --top-k 5
```

This creates:

- the `.onnx` model file
- a metadata `.json` file containing `label_names`, `feature_size`, and `sequence_length`

## 13. Replace the browser model

The current runtime points to:

- `models/asl_citizen_50.onnx`
- `models/asl_citizen_50_metadata.json`

You have two options:

### Option A. Replace the existing files

- keep the same names
- overwrite the old model and metadata

### Option B. Add a new model and update runtime references

Update the paths in:

- [`js/sign-model-runtime.js`](/D:/Integration-Game/gesture-trainer-web/js/sign-model-runtime.js)

This is cleaner if you want to keep multiple model versions.

## 14. Why letters should not be mixed into the first everyday model

Practical conclusion from the available data:

- alphabet classes have too few examples
- everyday words and letters are different recognition problems
- mixing them usually weakens the first model

Recommended strategy:

- model 1: everyday words
- model 2: letters or fingerspelling

## 15. Practical recommendations

### If time is limited

- do not jump straight to `100+` classes
- start with `20-50` useful signs
- inspect confusion pairs
- only then expand the vocabulary

### If classes have too few samples

- reduce the number of classes
- instead of keeping many highly similar classes with only about `30` videos each

### If browser inference feels weak

Check:

- whether the sequence length matches the metadata
- whether the ONNX file and metadata belong to the same model
- whether the runtime still points to an old model path
- whether the browser is using a cached build


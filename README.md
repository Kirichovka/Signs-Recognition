# Gesture Trainer Web

Technical documentation for the current browser-based ASL practice application.

## 1. Project Summary

This repository contains a static web application for ASL practice and a set of Python tools for data preparation, optional local APIs, model training, and export.

The **current primary runtime path** is:

- camera access in the browser
- hand landmark extraction with **MediaPipe Hands**
- gesture classification in JavaScript with a **weighted k-nearest neighbors (kNN)** matcher
- sample storage in [`datasets/landmarks_dataset.json`](./datasets/landmarks_dataset.json)

The main trainer does **not** require a Python backend to run. The site can be served as a static website from:

- GitHub Pages
- Netlify
- a local static server such as `python -m http.server`

## 2. Current Runtime Architecture

### Browser runtime

The current browser stack is:

- `HTML/CSS` for UI
- `MediaPipe Hands` for live hand landmark extraction
- `JavaScript` for:
  - landmark normalization
  - mirrored matching
  - weighted `kNN` voting
  - target/hold logic
  - dictionary and task flow

Main browser entry points:

- [`index.html`](./index.html)  
  Main trainer UI and dictionary
- [`model-test.html`](./model-test.html)  
  Low-level live matcher diagnostics
- [`alphabet-a-test.html`](./alphabet-a-test.html)  
  Task-driven JSON letter practice page

Main browser scripts:

- [`js/gesture-trainer.js`](./js/gesture-trainer.js)  
  Primary app logic
- [`js/model-test.js`](./js/model-test.js)  
  Diagnostic matcher page
- [`js/alphabet-a-test.js`](./js/alphabet-a-test.js)  
  Letter task practice over the JSON dataset
- [`js/sign-model-runtime.js`](./js/sign-model-runtime.js)  
  Shared camera/runtime utilities and legacy ONNX support

### Python runtime

Python is no longer required for the main static web experience, but it is still used for:

- dataset preparation
- duplicate checking
- training and export workflows
- optional local inference/API endpoints
- optional ONNX model export and legacy sequence-model tooling

Relevant Python files:

- [`python/local_inference_server.py`](./python/local_inference_server.py)
- [`python/train_sign_model.py`](./python/train_sign_model.py)
- [`python/export_sign_model_onnx.py`](./python/export_sign_model_onnx.py)
- [`python/run_word_model_pipeline.py`](./python/run_word_model_pipeline.py)
- [`python/run_alphabet_model_pipeline.py`](./python/run_alphabet_model_pipeline.py)
- [`python/run_model_pipelines.py`](./python/run_model_pipelines.py)

## 3. How Recognition Works

### 3.1 Main idea

The current recognizer is **not** a neural network running in the browser for the main trainer. It is a **landmark-based weighted kNN classifier**.

The browser:

1. captures a webcam frame
2. runs MediaPipe Hands
3. selects the primary detected hand
4. normalizes the `21` hand landmarks
5. compares them to saved samples in JSON
6. predicts the nearest label with weighted voting

### 3.2 Landmark preprocessing

For each live hand:

1. Choose the hand with the best `handedness score`.
2. Sort landmarks by `id`.
3. Recenter all points relative to landmark `0` (`wrist`).
4. Compute scale from the maximum `2D` distance from the wrist.
5. Normalize `x` and `y` by that scale.
6. Downweight depth with `Z_WEIGHT = 0.35`.
7. Flatten the result into a vector of `21 * 3 = 63` values.
8. Build a mirrored version by flipping `x`.

### 3.3 Distance and voting

For each saved sample:

1. Load its primary hand.
2. Normalize it with the same preprocessing pipeline.
3. Compute distance from the live vector to:
   - the original sample vector
   - the mirrored live vector
4. Keep the smaller distance.

Then:

1. Sort all samples by distance.
2. Keep the `K_NEIGHBORS = 5` nearest samples.
3. Weight each neighbor vote with:

```text
exp(-4 * distance)
```

4. Sum votes by label.
5. Pick the label with the highest total vote.

Derived outputs:

- `predicted_label`: label with the highest weighted vote
- `confidence`: winner vote share among the selected neighbors
- `similarity`: `1 / (1 + best_distance)`
- `top_matches`: top ranked labels and distances

### 3.4 Why this works on a static website

The site works without a backend because:

- MediaPipe runs in the browser
- the classifier is written in JavaScript
- training samples are stored as static JSON
- no server-side inference is required

In practice, the static site only needs to serve:

- page files
- JavaScript
- CSS
- `datasets/landmarks_dataset.json`
- optional media assets in [`videos`](./videos)

## 4. Dataset Format

The browser matcher reads:

- [`datasets/landmarks_dataset.json`](./datasets/landmarks_dataset.json)

High-level structure:

```json
{
  "labels": ["A", "B", "hello"],
  "samples": [
    {
      "id": 1,
      "label": "A",
      "captured_at": "2026-03-19T00:00:00Z",
      "image_width": 960,
      "image_height": 540,
      "hands": [
        {
          "handedness": "Right",
          "score": 0.99,
          "image_landmarks": [
            { "id": 0, "x": 0.5, "y": 0.6, "z": 0.0 }
          ],
          "world_landmarks": []
        }
      ]
    }
  ]
}
```

Important implementation notes:

- only samples that contain a hand with at least `21` `image_landmarks` are usable
- the matcher always selects the best-scored hand from each sample
- labels listed in `labels` are not enough by themselves; a label must also have at least one valid sample in `samples`

Current repository state:

- the dataset file declares more labels than it currently has usable samples for
- the current usable sample labels are primarily:
  - letters: `A, B, C, D, E, F, G, H, I, J, L`
  - words: `hello`

This means:

- letter practice is broader than word practice
- the word matcher is currently limited by dataset coverage, not by code

## 5. Application Pages

### 5.1 Main trainer

File:

- [`index.html`](./index.html)

Behavior:

- landing page
- dictionary page
- alphabet page
- practice page
- camera overlay with live landmarks
- target sign tasks
- hold progress
- speech playback for the current target
- fallback descriptions and media for dictionary items

Recognition backend:

- JSON landmark matcher from [`datasets/landmarks_dataset.json`](./datasets/landmarks_dataset.json)

### 5.2 Model test page

File:

- [`model-test.html`](./model-test.html)

Purpose:

- raw live recognition diagnostics
- top prediction inspection
- confidence/similarity inspection
- dataset loading and camera troubleshooting

Recognition backend:

- same JSON kNN matcher as the main trainer

### 5.3 Letter practice page

File:

- [`alphabet-a-test.html`](./alphabet-a-test.html)

Purpose:

- task-oriented letter practice
- current task such as "Show A"
- hold progress
- confidence/similarity display
- top matches and diagnostics

Recognition backend:

- same JSON kNN matcher

Despite the historical file name, this page is no longer limited to only one hardcoded letter.

## 6. Dictionary and Media Assets

The visual dictionary in the main trainer uses a curated set of local media files from:

- [`videos`](./videos)

These assets are used for:

- dictionary cards
- popup demonstrations
- sign descriptions

They are **not** the recognition source. Recognition comes from the JSON landmark dataset, not from the video files.

The app deliberately avoids probing missing local media with network `HEAD` requests and instead uses an explicit allowlist in the frontend to prevent noisy `404` errors.

## 7. How Python and JavaScript Interact

Python libraries are **not imported directly into JavaScript**.

Instead, the project uses two patterns:

### Pattern A: export artifacts for browser use

Python is used to train or prepare a model, then export artifacts such as:

- `.onnx`
- metadata `.json`

The browser then loads those exported files directly.

Example:

- exporter: [`python/export_sign_model_onnx.py`](./python/export_sign_model_onnx.py)
- browser runtime: [`js/sign-model-runtime.js`](./js/sign-model-runtime.js)

### Pattern B: port the algorithm from Python to JavaScript

For the current JSON matcher, the algorithm exists in Python as an optional service and was then reimplemented in JavaScript for static hosting.

Python side:

- [`python/local_inference_server.py`](./python/local_inference_server.py)

Browser side:

- [`js/gesture-trainer.js`](./js/gesture-trainer.js)
- [`js/model-test.js`](./js/model-test.js)
- [`js/alphabet-a-test.js`](./js/alphabet-a-test.js)

This is why the app can use Python during development, but still run entirely as a static site in production.

## 8. Optional Local API

The repository still includes an optional FastAPI service:

- [`python/local_inference_server.py`](./python/local_inference_server.py)

Available endpoints:

- `GET /`
- `GET /api/health`
- `GET /api/landmarks/stats`
- `GET /api/landmarks/labels`
- `POST /api/landmarks/reload`
- `POST /api/recognize/landmarks`
- `POST /api/predict`
- `POST /api/recognize/sequence`

Current role of this service:

- local debugging
- optional server-side landmark matching
- optional legacy sequence-model inference

It is **not required** for the GitHub Pages deployment.

## 9. Legacy ONNX / Sequence Model Infrastructure

The repository still contains legacy infrastructure for neural-network-based recognition:

- exported models in [`models`](./models)
- ONNX runtime support in [`js/sign-model-runtime.js`](./js/sign-model-runtime.js)
- training/export utilities in [`python`](./python)

These files remain useful for:

- experiments
- comparisons
- future hybrid systems

But the **current main trainer path** is JSON landmark matching, not ONNX inference.

## 10. Local Development

### Static site

Use a local HTTP server. Camera access is not reliable from `file://`.

Example:

```powershell
cd D:\Integration-Game\gesture-trainer-web
python -m http.server 4174
```

Open:

- `http://127.0.0.1:4174/`
- `http://127.0.0.1:4174/model-test.html`
- `http://127.0.0.1:4174/alphabet-a-test.html`

### Python environment

If you need training or the optional API:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 11. Project Structure

```text
gesture-trainer-web/
  index.html
  model-test.html
  alphabet-a-test.html
  styles.css
  hero.png
  datasets/
    landmarks_dataset.json
  videos/
  js/
    gesture-trainer.js
    model-test.js
    alphabet-a-test.js
    sign-model-runtime.js
  models/
    *.onnx
    *_metadata.json
  python/
    local_inference_server.py
    train_sign_model.py
    export_sign_model_onnx.py
    run_word_model_pipeline.py
    run_alphabet_model_pipeline.py
    run_model_pipelines.py
  docs/
    ARCHITECTURE.md
    TRAINING_PIPELINE.md
    WORD_MODEL_PIPELINE.md
    ALPHABET_MODEL_PIPELINE.md
    WEB_EXPORT.md
    TROUBLESHOOTING.md
```

## 12. Current Limitations

- The browser matcher is only as good as the sample coverage in `landmarks_dataset.json`.
- Word-level recognition is currently limited because the dataset contains very few usable word samples.
- The matcher uses a single live hand and does not model temporal motion.
- Similar handshapes can still be confused, especially with small datasets.
- Some legacy ONNX files and documentation remain in the repository for experimentation, even though they are not the main runtime path anymore.

## 13. Recommended Next Steps

If you want to improve recognition quality, the most effective next steps are:

1. add more labeled samples to [`datasets/landmarks_dataset.json`](./datasets/landmarks_dataset.json)
2. expand word-level sample coverage beyond `hello`
3. keep consistent camera framing when collecting samples
4. optionally add a server-side evaluation workflow through [`python/local_inference_server.py`](./python/local_inference_server.py)
5. optionally revisit ONNX models later for hybrid or temporal recognition

## 14. Related Documentation

For deeper workflow documentation, see:

- [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)
- [`docs/TRAINING_PIPELINE.md`](./docs/TRAINING_PIPELINE.md)
- [`docs/WORD_MODEL_PIPELINE.md`](./docs/WORD_MODEL_PIPELINE.md)
- [`docs/ALPHABET_MODEL_PIPELINE.md`](./docs/ALPHABET_MODEL_PIPELINE.md)
- [`docs/MS_ASL_AUGMENTATION.md`](./docs/MS_ASL_AUGMENTATION.md)
- [`docs/WEB_EXPORT.md`](./docs/WEB_EXPORT.md)
- [`docs/TROUBLESHOOTING.md`](./docs/TROUBLESHOOTING.md)

# ASL Browser Trainer

Interactive sign-practice project built with MediaPipe Holistic and `onnxruntime-web`.

The current web UI no longer depends on a Python backend. The trained sign model now runs directly in the browser, which means the app can be hosted as a static site on GitHub Pages, Netlify, or a local `http.server`.

## Current features

- browser-based sign trainer with target signs, hold progress, coaching, and camera diagnostics
- standalone live model test page
- ONNX model stored in the repository
- Python training pipeline for manifests, subset preparation, landmark feature extraction, GRU training, and ONNX export
- curated label set for practical everyday ASL signs

Current embedded model:

- model file: [`models/asl_citizen_50.onnx`](/D:/Integration-Game/gesture-trainer-web/models/asl_citizen_50.onnx)
- metadata: [`models/asl_citizen_50_metadata.json`](/D:/Integration-Game/gesture-trainer-web/models/asl_citizen_50_metadata.json)
- source: baseline trained on an `ASL Citizen` subset with `50` classes

## Quick start

### Web version

```powershell
cd D:\Integration-Game\gesture-trainer-web
python -m http.server 4174
```

Open:

- `http://127.0.0.1:4174/`
- `http://127.0.0.1:4174/model-test.html`

### Which page to use

- `index.html` - main trainer with target signs, hold logic, coaching, and diagnostics
- `model-test.html` - lightweight page for camera checks and raw top predictions

## Current architecture

### In the browser

- `index.html` and `model-test.html` load:
  - `onnxruntime-web`
  - MediaPipe Holistic
  - local JS modules from [`js`](/D:/Integration-Game/gesture-trainer-web/js)
- camera capture and landmark extraction run on the client
- a `40`-frame landmark sequence is sent into the ONNX model
- top predictions are computed directly in the browser

### In Python

Python is now used for:

- dataset manifest creation
- subset selection from explicit label lists
- video feature extraction
- model training
- exporting PyTorch checkpoints to ONNX

The local FastAPI backend is still included as a helper tool, but it is no longer required for the browser UI.

## Project structure

```text
gesture-trainer-web/
  index.html
  model-test.html
  styles.css
  js/
    gesture-trainer.js
    model-test.js
    sign-model-runtime.js
  models/
    asl_citizen_50.onnx
    asl_citizen_50_metadata.json
  python/
    build_asl_citizen_manifest.py
    build_wlasl_manifest.py
    prepare_wlasl_subset.py
    extract_sign_features.py
    train_sign_model.py
    export_sign_model_onnx.py
    local_inference_server.py
    label_sets/
      asl_citizen_daily_v1.txt
  docs/
    ARCHITECTURE.md
    TRAINING_PIPELINE.md
    TROUBLESHOOTING.md
```

## Key files

- [`js/sign-model-runtime.js`](/D:/Integration-Game/gesture-trainer-web/js/sign-model-runtime.js)  
  Shared browser runtime for ONNX loading, softmax, landmark normalization, feature-vector generation, inference, and camera startup.

- [`js/gesture-trainer.js`](/D:/Integration-Game/gesture-trainer-web/js/gesture-trainer.js)  
  Main trainer UI: target-sign flow, hold progress, visibility coaching, zone checking, guide card, and diagnostics.

- [`js/model-test.js`](/D:/Integration-Game/gesture-trainer-web/js/model-test.js)  
  Minimal live test page without the full training loop.

- [`python/train_sign_model.py`](/D:/Integration-Game/gesture-trainer-web/python/train_sign_model.py)  
  GRU-based classifier training on landmark sequences.

- [`python/export_sign_model_onnx.py`](/D:/Integration-Game/gesture-trainer-web/python/export_sign_model_onnx.py)  
  Export utility for converting a trained checkpoint into ONNX and browser metadata.

- [`python/label_sets/asl_citizen_daily_v1.txt`](/D:/Integration-Game/gesture-trainer-web/python/label_sets/asl_citizen_daily_v1.txt)  
  Curated list of practical everyday ASL labels for the next training run.

## Current model status

Baseline result that has already been trained:

- dataset: `ASL Citizen`
- subset size: `50` classes
- total videos: `1901`
- extracted feature shape: `(1901, 40, 154)`
- baseline validation accuracy: about `56.35%`

What that means:

- the pipeline works end to end
- the model is good enough for a prototype and demo baseline
- the quality is still below production-grade recognition
- a curated everyday subset and/or fewer conflicting classes should improve usability

## Curated everyday labels

Prepared label set for the next training run:

- [`python/label_sets/asl_citizen_daily_v1.txt`](/D:/Integration-Game/gesture-trainer-web/python/label_sets/asl_citizen_daily_v1.txt)

It contains `40` practical signs:

- `HELLO`
- `BYE`
- `YES`
- `NO`
- `PLEASE`
- `SORRY`
- `HELP`
- `THANKYOU`
- `WELCOME1`
- `EAT1`
- `DRINK1`
- `WATER`
- `MOTHER`
- `FATHER`
- `FAMILY`
- `HOME`
- `HOUSE`
- `SCHOOL`
- `WORK`
- `FRIEND`
- `LOVE`
- `WANT1`
- `NEED`
- `COME`
- `COMEHERE`
- `GO`
- `STOP`
- `FINISH`
- `GOOD`
- `BAD`
- `HAPPY`
- `SAD`
- `NOW`
- `MORE`
- `NOT`
- `KNOW`
- `DONTKNOW`
- `NOTUNDERSTAND`
- `GOAHEAD`
- `GREAT`

Important note:

- letters are intentionally not mixed into this model yet
- alphabet classes currently have too few examples in the available data, so letter recognition should be trained separately

## Alphabet dataset path

For a separate alphabet-only model, the repository now includes a dedicated download path for a static image dataset:

- guide: [Alphabet Dataset](./docs/ALPHABET_DATASET.md)
- downloader: [`python/download_asl_semcom.py`](/D:/Integration-Game/gesture-trainer-web/python/download_asl_semcom.py)

Recommended alphabet dataset:

- **ASL_SemCom** from Zenodo: [dataset page](https://zenodo.org/records/14635573)

Quick start:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python download_asl_semcom.py
```

This path is intentionally separate from the word-level pipeline:

- word model = video / temporal landmarks
- alphabet model = static image classification

For duplicate checks in alphabet image datasets:

- scanner: [`python/check_image_duplicates.py`](/D:/Integration-Game/gesture-trainer-web/python/check_image_duplicates.py)
- docs: [Alphabet Dataset](./docs/ALPHABET_DATASET.md)

## Extra word videos from a second dataset

The repository also now includes a merge workflow for adding **extra word-level training videos** from a second dataset such as **MS-ASL**.

New resources:

- guide: [MS-ASL Augmentation](./docs/MS_ASL_AUGMENTATION.md)
- merge utility: [`python/merge_sign_manifests.py`](/D:/Integration-Game/gesture-trainer-web/python/merge_sign_manifests.py)

Recommended approach:

- keep `ASL Citizen` as the primary dataset
- use `MS-ASL` as extra **train-only** augmentation
- keep validation and test behavior anchored in the base dataset

## Combined dataset bootstrap and two-model layout

The repository also now includes a single bootstrap script for preparing the dataset workspace and the output folders for two separate models:

- bootstrap script: [`python/download_sign_datasets.py`](/D:/Integration-Game/gesture-trainer-web/python/download_sign_datasets.py)
- workflow guide: [Dual Model Workflow](./docs/DUAL_MODEL_WORKFLOW.md)

Quick start:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python download_sign_datasets.py
```

This sets up:

- `ASL Citizen` for the word model
- `MS-ASL` as a second word-level source
- `ASL_SemCom` for the alphabet model
- separate artifact folders for:
  - `artifacts/word_model`
  - `artifacts/alphabet_model`

## One-command word model pipeline

The repository now also includes an orchestration script for the full word-model training path:

- script: [`python/run_word_model_pipeline.py`](/D:/Integration-Game/gesture-trainer-web/python/run_word_model_pipeline.py)
- guide: [Word Model Pipeline](./docs/WORD_MODEL_PIPELINE.md)

It can:

- build the full `ASL Citizen` manifest
- prepare the curated everyday subset
- optionally merge an extra `MS-ASL` manifest
- run duplicate video checks on the final manifest
- run feature extraction
- train the GRU model
- export ONNX

Sample `MS-ASL` label map:

- [`python/label_maps/ms_asl_daily_map.example.csv`](/D:/Integration-Game/gesture-trainer-web/python/label_maps/ms_asl_daily_map.example.csv)

For duplicate and leakage checks in video datasets:

- scanner: [`python/check_video_duplicates.py`](/D:/Integration-Game/gesture-trainer-web/python/check_video_duplicates.py)

## One-command alphabet model pipeline

The repository now also includes a dedicated alphabet-model orchestration path:

- script: [`python/run_alphabet_model_pipeline.py`](/D:/Integration-Game/gesture-trainer-web/python/run_alphabet_model_pipeline.py)
- guide: [Alphabet Model Pipeline](./docs/ALPHABET_MODEL_PIPELINE.md)

It can:

- run duplicate image checks
- train a static alphabet classifier
- export ONNX

## Run one or both pipelines from a single launcher

Top-level launcher:

- [`python/run_model_pipelines.py`](/D:/Integration-Game/gesture-trainer-web/python/run_model_pipelines.py)

Examples:

Run both:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python run_model_pipelines.py ^
  --mode all ^
  --word-dataset-root D:\Integration-Game\gesture-trainer-web\datasets\asl_citizen\ASL_Citizen ^
  --alphabet-dataset-root D:\Integration-Game\gesture-trainer-web\datasets\asl_semcom\ASL_SemCom
```

Run only the word model:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python run_model_pipelines.py ^
  --mode word ^
  --word-dataset-root D:\Integration-Game\gesture-trainer-web\datasets\asl_citizen\ASL_Citizen
```

Run only the alphabet model:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python run_model_pipelines.py ^
  --mode alphabet ^
  --alphabet-dataset-root D:\Integration-Game\gesture-trainer-web\datasets\asl_semcom\ASL_SemCom
```

## Optional local Python backend

Even though the web app now works without a backend, the local FastAPI server is still useful for:

- smoke-testing an old API flow
- debugging a raw PyTorch checkpoint before ONNX export
- checking `/api/health` and `/api/predict` locally

Example:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn local_inference_server:app --host 127.0.0.1 --port 8000
```

That server is optional for normal website use.

## Detailed documentation

- [Architecture](./docs/ARCHITECTURE.md)
- [Training Pipeline](./docs/TRAINING_PIPELINE.md)
- [Troubleshooting](./docs/TROUBLESHOOTING.md)

## Short retraining workflow

1. Download and extract `ASL Citizen`
2. Build the full manifest
3. Create a subset with `prepare_wlasl_subset.py`
4. Extract features with `extract_sign_features.py`
5. Train a model with `train_sign_model.py`
6. Export the checkpoint to ONNX with `export_sign_model_onnx.py`
7. Replace the files in `models/`
8. Reload the static website

Full commands and explanations are in [Training Pipeline](./docs/TRAINING_PIPELINE.md).

## Known limitations

- the current embedded model is still a baseline
- many visually similar signs can still be confused
- recognition quality depends heavily on stable visibility of hands, upper body, and face
- a words-only model should not be expected to handle alphabet recognition well
- GitHub Pages and Netlify are suitable for the browser ONNX version, but not for Python inference

## Deployment

Any static host is fine:

- GitHub Pages
- Netlify
- Vercel static hosting
- local `python -m http.server`

Requirements:

- the site must be able to serve the files in `models/`
- the browser must be allowed to access the camera
- CDN dependencies for `onnxruntime-web` and MediaPipe Holistic must load correctly

## Useful notes

- if you see `404` on `/api/health`, you are probably opening an old backend-oriented page or a cached build
- if the camera throws `NotFoundError`, check browser permissions, the selected device, and whether another app is using the camera
- if Python feature extraction fails with `libGL.so.1`, install the `libgl1` system package on Linux

# Dual Model Workflow

## Goal

This document describes the recommended workflow for maintaining **two separate models** in the project:

- a **word model**
- an **alphabet model**

This is the cleanest architecture for the current project.

## Why two models are better than one combined model

The project currently mixes two different recognition problems:

1. **word-level isolated sign recognition**
2. **alphabet-only static handshape recognition**

They should not be treated as the same problem.

Why:

- the word model is temporal and depends on motion over time
- the alphabet model can be mostly static image classification
- some letters like `J` and `Z` require motion, while most letters do not
- combining everything into one early model usually makes training and debugging harder

Recommended split:

- **word model**
  - datasets: `ASL Citizen`, optionally `MS-ASL`
  - input type: video -> landmarks -> sequence model
  - target task: practical ASL words

- **alphabet model**
  - dataset: `ASL_SemCom`
  - input type: static image classifier
  - target task: alphabet handshapes

## Repository support

The repository now includes a bootstrap script:

- [`python/download_sign_datasets.py`](/D:/Integration-Game/gesture-trainer-web/python/download_sign_datasets.py)

It can:

- download `ASL Citizen`
- download `MS-ASL`
- download `ASL_SemCom`
- extract archives
- prepare separate artifact directories for:
  - `artifacts/word_model`
  - `artifacts/alphabet_model`

## One-command dataset bootstrap

From the repository root:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python download_sign_datasets.py
```

This prepares:

- [`datasets/asl_citizen`](/D:/Integration-Game/gesture-trainer-web/datasets/asl_citizen)
- [`datasets/ms_asl`](/D:/Integration-Game/gesture-trainer-web/datasets/ms_asl)
- [`datasets/asl_semcom`](/D:/Integration-Game/gesture-trainer-web/datasets/asl_semcom)
- [`artifacts/word_model`](/D:/Integration-Game/gesture-trainer-web/artifacts/word_model)
- [`artifacts/alphabet_model`](/D:/Integration-Game/gesture-trainer-web/artifacts/alphabet_model)

## Download only

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python download_sign_datasets.py --download-only
```

## Skip selected datasets

Example:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python download_sign_datasets.py --skip-ms-asl
```

## Word model path

Recommended word-model pipeline:

1. Download and extract `ASL Citizen`
2. Build the full `ASL Citizen` manifest
3. Build the curated everyday subset
4. Optionally add extra train rows from `MS-ASL`
5. Extract features
6. Train the GRU word model
7. Export ONNX if needed
8. Save outputs into:
   - [`artifacts/word_model`](/D:/Integration-Game/gesture-trainer-web/artifacts/word_model)

Recommended output files:

- `best_model.pt`
- `training_metrics.json`
- `everyday_words.onnx`
- `everyday_words_metadata.json`

## Alphabet model path

Recommended alphabet-model pipeline:

1. Download and extract `ASL_SemCom`
2. Inspect folder structure
3. Build an image classification dataset loader
4. Train a separate alphabet classifier
5. Save outputs into:
   - [`artifacts/alphabet_model`](/D:/Integration-Game/gesture-trainer-web/artifacts/alphabet_model)

Recommended output files:

- `alphabet_best_model.pt`
- `alphabet_training_metrics.json`
- `alphabet.onnx`
- `alphabet_metadata.json`

## Important note about MS-ASL

The official `MS-ASL.zip` package is an annotation package, not a full extracted clip archive.

The repository now handles this by:

- reading the annotation files
- downloading needed source videos with `yt-dlp`
- clipping local samples only for the overlapping everyday labels
- building a local manifest from those prepared clips

## Suggested next step

The project is now ready for:

1. a combined dataset bootstrap
2. a separate word-model training path
3. a separate alphabet-model training path

There is now also a dedicated orchestration path for the word model:

- [Word Model Pipeline](./WORD_MODEL_PIPELINE.md)

And a dedicated orchestration path for the alphabet model:

- [Alphabet Model Pipeline](./ALPHABET_MODEL_PIPELINE.md)

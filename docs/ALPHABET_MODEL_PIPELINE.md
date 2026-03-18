# Alphabet Model Pipeline

## Goal

This document describes the dedicated orchestration path for the **alphabet model**.

The alphabet model is intentionally separate from the word model:

- word model = temporal sign recognition
- alphabet model = static image classification

## Main script

- [`python/run_alphabet_model_pipeline.py`](/D:/Integration-Game/gesture-trainer-web/python/run_alphabet_model_pipeline.py)

It runs:

1. duplicate image scanning
2. alphabet model training
3. optional ONNX export

## Training script

- [`python/train_alphabet_model.py`](/D:/Integration-Game/gesture-trainer-web/python/train_alphabet_model.py)

This script expects a dataset root with split folders such as:

```text
ASL_SemCom/
  train/
    A/
    B/
    ...
  test/
    A/
    B/
    ...
```

## ONNX export script

- [`python/export_alphabet_model_onnx.py`](/D:/Integration-Game/gesture-trainer-web/python/export_alphabet_model_onnx.py)

There is also a unified web-export launcher:

- [Web Export](./WEB_EXPORT.md)

## Example run

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python run_alphabet_model_pipeline.py ^
  --dataset-root D:\Integration-Game\gesture-trainer-web\datasets\asl_semcom\ASL_SemCom ^
  --run-name alphabet_v1
```

## What the pipeline writes

Outputs go to:

- [`artifacts/alphabet_model`](/D:/Integration-Game/gesture-trainer-web/artifacts/alphabet_model)

Example run folder:

```text
artifacts/
  alphabet_model/
    alphabet_v1/
      alphabet_v1_duplicate_report.json
      alphabet_v1.onnx
      alphabet_v1_metadata.json
      pipeline_summary.json
      training_run/
        best_model.pt
        training_metrics.json
```

## Duplicate safety

The pipeline automatically runs:

- [`python/check_image_duplicates.py`](/D:/Integration-Game/gesture-trainer-web/python/check_image_duplicates.py)

By default it fails if exact duplicates are found across train/test-like splits.

Flags:

- `--skip-duplicate-check`
- `--allow-cross-split-duplicates`

## Current scope

The current alphabet path is designed for:

- static handshape letters
- RGB image classification
- the `ASL_SemCom` dataset

Motion letters such as `J` and `Z` still need a separate temporal treatment later.

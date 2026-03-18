# Web Export

## Goal

This document describes the unified export path for converting trained models into browser-ready ONNX artifacts.

The repository already contains separate exporters for:

- word models
- alphabet models

There is now also a unified launcher that can export:

- the word model
- the alphabet model
- or both in one run

## Main script

- [`python/export_models_for_web.py`](/D:/Integration-Game/gesture-trainer-web/python/export_models_for_web.py)

This script wraps:

- [`python/export_sign_model_onnx.py`](/D:/Integration-Game/gesture-trainer-web/python/export_sign_model_onnx.py)
- [`python/export_alphabet_model_onnx.py`](/D:/Integration-Game/gesture-trainer-web/python/export_alphabet_model_onnx.py)

## What it produces

For each selected model it generates:

- `.onnx`
- metadata `.json`

By default these files are written under:

- [`artifacts`](/D:/Integration-Game/gesture-trainer-web/artifacts)

Layout:

```text
artifacts/
  web_exports/
    word/
      word_model.onnx
      word_model_metadata.json
    alphabet/
      alphabet_model.onnx
      alphabet_model_metadata.json
```

## Export only the word model

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python export_models_for_web.py ^
  --mode word ^
  --word-checkpoint D:\Integration-Game\gesture-trainer-web\artifacts\word_model\everyday_daily_v1\training_run\best_model.pt ^
  --word-name everyday_daily_v1
```

## Export only the alphabet model

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python export_models_for_web.py ^
  --mode alphabet ^
  --alphabet-checkpoint D:\Integration-Game\gesture-trainer-web\artifacts\alphabet_model\alphabet_v1\training_run\best_model.pt ^
  --alphabet-name alphabet_v1
```

## Export both models in one run

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python export_models_for_web.py ^
  --mode all ^
  --word-checkpoint D:\Integration-Game\gesture-trainer-web\artifacts\word_model\everyday_daily_v1\training_run\best_model.pt ^
  --alphabet-checkpoint D:\Integration-Game\gesture-trainer-web\artifacts\alphabet_model\alphabet_v1\training_run\best_model.pt ^
  --word-name everyday_daily_v1 ^
  --alphabet-name alphabet_v1
```

## Publish directly into the web `models/` folder

If you want the exported artifacts copied directly into:

- [`models`](/D:/Integration-Game/gesture-trainer-web/models)

Use:

- `--publish-word`
- `--publish-alphabet`

Example:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python export_models_for_web.py ^
  --mode all ^
  --word-checkpoint D:\Integration-Game\gesture-trainer-web\artifacts\word_model\everyday_daily_v1\training_run\best_model.pt ^
  --alphabet-checkpoint D:\Integration-Game\gesture-trainer-web\artifacts\alphabet_model\alphabet_v1\training_run\best_model.pt ^
  --word-name everyday_daily_v1 ^
  --alphabet-name alphabet_v1 ^
  --publish-word ^
  --publish-alphabet
```

## Important note

Publishing both models into the same `models/` folder is useful only if the frontend is prepared to select between them.

If the web runtime still points to one fixed model path, publishing multiple ONNX files is safe, but the UI will not switch between them automatically until the frontend is updated to do so.

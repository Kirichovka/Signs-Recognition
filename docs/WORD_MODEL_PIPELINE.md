# Word Model Pipeline

## Goal

This document describes the one-command orchestration path for the **word model**.

The script covers the main sequence:

1. build the full `ASL Citizen` manifest
2. create the curated subset
3. optionally merge extra `MS-ASL` rows
4. extract landmark features
5. train the GRU model
6. export ONNX

## Pipeline script

Main orchestration script:

- [`python/run_word_model_pipeline.py`](/D:/Integration-Game/gesture-trainer-web/python/run_word_model_pipeline.py)

The script is designed so that:

- `ASL Citizen` is the base dataset
- `MS-ASL` is optional extra training data
- all outputs are grouped into one run folder

## Default curated run

If you want to train the curated everyday word model from `ASL Citizen` only:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python run_word_model_pipeline.py ^
  --dataset-root D:\Integration-Game\gesture-trainer-web\datasets\asl_citizen\ASL_Citizen ^
  --run-name everyday_daily_v1
```

Default label list:

- [`python/label_sets/asl_citizen_daily_v1.txt`](/D:/Integration-Game/gesture-trainer-web/python/label_sets/asl_citizen_daily_v1.txt)

## Run with MS-ASL augmentation

If you already have an `MS-ASL` overlap manifest:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python run_word_model_pipeline.py ^
  --dataset-root D:\Integration-Game\gesture-trainer-web\datasets\asl_citizen\ASL_Citizen ^
  --run-name everyday_daily_v1_merged ^
  --extra-manifest D:\Integration-Game\gesture-trainer-web\datasets\ms_asl\ms_asl_daily_overlap_manifest.jsonl ^
  --extra-label-map D:\Integration-Game\gesture-trainer-web\python\label_maps\ms_asl_daily_map.example.csv
```

Sample label map:

- [`python/label_maps/ms_asl_daily_map.example.csv`](/D:/Integration-Game/gesture-trainer-web/python/label_maps/ms_asl_daily_map.example.csv)

## What the script writes

By default, outputs go to:

- [`artifacts/word_model`](/D:/Integration-Game/gesture-trainer-web/artifacts/word_model)

Each run gets its own folder:

```text
artifacts/
  word_model/
    everyday_daily_v1/
      everyday_daily_v1_asl_citizen_manifest.jsonl
      everyday_daily_v1_subset_manifest.jsonl
      everyday_daily_v1_subset_stats.json
      everyday_daily_v1_features.npz
      everyday_daily_v1.onnx
      everyday_daily_v1_metadata.json
      pipeline_summary.json
      training_run/
        best_model.pt
        training_metrics.json
```

If `MS-ASL` augmentation is enabled, the run folder also contains:

- merged manifest
- merge stats

## Main options

### Basic inputs

- `--dataset-root`  
  Path to the extracted `ASL_Citizen` dataset root.

- `--run-name`  
  Name used for the output folder and generated files.

- `--labels-file`  
  Optional custom label list. Defaults to the curated everyday file.

### Optional extra dataset

- `--extra-manifest`  
  Manifest for extra training rows, such as `MS-ASL`.

- `--extra-label-map`  
  CSV mapping labels from the extra dataset into the base label space.

### Subset controls

- `--max-train-per-class`
- `--max-val-per-class`

### Feature extraction

- `--max-frames`

### Training

- `--epochs`
- `--batch-size`
- `--hidden-size`
- `--dropout`
- `--lr`
- `--seed`

### Export

- `--skip-export`

### Re-running

- `--force`  
  Rebuild intermediate artifacts even if they already exist.

## Why this script is useful

Without orchestration, the word-model pipeline is easy to break across multiple manual commands.

This script gives:

- one run folder per training attempt
- reproducible paths
- optional merge with a second dataset
- automatic ONNX export
- a final `pipeline_summary.json`

## Current limitation

The script assumes:

- the `ASL Citizen` dataset root is already extracted
- an `MS-ASL` manifest already exists if you pass `--extra-manifest`

So the current state is:

- `ASL Citizen` path is end-to-end ready
- `MS-ASL` path is ready from the merge step onward
- the missing piece for `MS-ASL` is still the exact manifest builder after package inspection

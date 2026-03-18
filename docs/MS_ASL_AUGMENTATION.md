# MS-ASL Augmentation

## Goal

This document describes how to use **MS-ASL** as a second source of **additional training videos** for the everyday word set.

The recommended strategy is:

- keep **ASL Citizen** as the primary dataset
- use **MS-ASL** as extra **train-only augmentation**
- keep validation and test behavior anchored in the base dataset

That gives a cleaner benchmark and avoids mixing evaluation standards.

## Why use MS-ASL as the second dataset

`MS-ASL` is a strong match for this role because it is:

- ASL, not a different signed language
- word-level / isolated-sign oriented
- large enough to contribute useful extra samples

Microsoft Research describes `MS-ASL` as a large-scale ASL dataset with:

- over `25,000` annotated videos
- `1000` signs
- `200+` signers

Sources:

- [MS-ASL project page](https://www.microsoft.com/en-us/research/project/ms-asl/)
- [MS-ASL paper](https://www.microsoft.com/applied-sciences/uploads/publications/3/ms-asl.pdf)

## Important practical note

The official Microsoft download page lists:

- file name: `MS-ASL.zip`
- file size: `1.9 MB`

Source:

- [Official download page](https://www.microsoft.com/en-us/download/details.aspx?id=100121)

That strongly suggests the official package is **not a full self-contained video archive**. In practice, treat it as an annotation package that still needs local clip preparation.

## Recommended merge strategy

Use this workflow:

1. Build the main subset from `ASL Citizen`
2. Build or prepare an `MS-ASL` manifest
3. Map `MS-ASL` labels to the same target labels as the ASL Citizen subset
4. Merge `MS-ASL` rows into the training split only
5. Keep evaluation focused on the base dataset

## Repository support

The repository now includes:

- [`python/merge_sign_manifests.py`](/D:/Integration-Game/gesture-trainer-web/python/merge_sign_manifests.py)
- [`python/download_ms_asl_clips.py`](/D:/Integration-Game/gesture-trainer-web/python/download_ms_asl_clips.py)
- [`python/build_ms_asl_manifest.py`](/D:/Integration-Game/gesture-trainer-web/python/build_ms_asl_manifest.py)

Current support can:

- merge a base manifest with an extra manifest
- automatically prepare overlapping local MS-ASL clips
- automatically build a local MS-ASL manifest from those clips
- remap labels from the extra dataset into the curated target label space
- force all extra rows into `train`
- keep only an allowed list of labels
- write merge statistics

## Intended setup

### Base dataset

Use `ASL Citizen` for:

- subset selection
- primary label definitions
- validation behavior

### Extra dataset

Use `MS-ASL` for:

- extra train samples only
- overlapping everyday words

## Step 1. Build the base everyday subset

Example:

```bash
python prepare_wlasl_subset.py \
  --manifest ~/workspace/datasets/asl_citizen/asl_citizen_manifest.jsonl \
  --labels-file ~/workspace/Signs-Recognition/python/label_sets/asl_citizen_daily_v1.txt \
  --output ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_manifest.jsonl \
  --stats-output ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_stats.json \
  --max-train-per-class 120 \
  --max-val-per-class 30
```

## Step 2. Prepare local MS-ASL clips

The repository can now do this automatically.

Standalone example:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python download_ms_asl_clips.py ^
  --dataset-root D:\Integration-Game\gesture-trainer-web\datasets\ms_asl\MS-ASL ^
  --output-root D:\Integration-Game\gesture-trainer-web\datasets\ms_asl\clips_daily_v1 ^
  --labels-file D:\Integration-Game\gesture-trainer-web\python\label_sets\asl_citizen_daily_v1.txt ^
  --splits train ^
  --max-clips-per-label 40
```

This script:

1. reads the MS-ASL annotation package
2. keeps only labels that overlap the curated everyday set
3. uses synonym groups to map variants like `dad` -> `FATHER` when possible
4. downloads source videos with `yt-dlp`
5. clips the requested local samples into `.mp4` files

## Step 3. Build an MS-ASL manifest

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python build_ms_asl_manifest.py ^
  --dataset-root D:\Integration-Game\gesture-trainer-web\datasets\ms_asl\MS-ASL ^
  --clips-root D:\Integration-Game\gesture-trainer-web\datasets\ms_asl\clips_daily_v1 ^
  --labels-file D:\Integration-Game\gesture-trainer-web\python\label_sets\asl_citizen_daily_v1.txt ^
  --splits train ^
  --output D:\Integration-Game\gesture-trainer-web\datasets\ms_asl\ms_asl_daily_v1_manifest.jsonl
```

## Step 4. Merge the manifests

If you want the manual merge path, you can still do this explicitly:

```bash
python merge_sign_manifests.py \
  --base-manifest ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_manifest.jsonl \
  --extra-manifest ~/workspace/datasets/ms_asl/ms_asl_daily_v1_manifest.jsonl \
  --allowed-labels-file ~/workspace/Signs-Recognition/python/label_sets/asl_citizen_daily_v1.txt \
  --force-extra-split train \
  --extra-source-name ms_asl \
  --base-source-name asl_citizen \
  --output ~/workspace/datasets/merged/everyday_daily_v1_merged_manifest.jsonl \
  --stats-output ~/workspace/datasets/merged/everyday_daily_v1_merged_stats.json
```

## Step 5. Use the fully automatic path

The easiest path is now:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python run_word_model_pipeline.py ^
  --dataset-root D:\Integration-Game\gesture-trainer-web\datasets\asl_citizen\ASL_Citizen ^
  --run-name everyday_daily_v1 ^
  --auto-ms-asl-root D:\Integration-Game\gesture-trainer-web\datasets\ms_asl\MS-ASL
```

This one command now:

1. builds the ASL Citizen manifest
2. prepares the curated everyday subset
3. downloads and clips overlapping MS-ASL train samples
4. builds the local MS-ASL manifest
5. merges it into train
6. runs duplicate checks
7. extracts features
8. trains the GRU
9. exports ONNX

## Duplicate video check

The repository now includes a duplicate scanner for videos:

- [`python/check_video_duplicates.py`](/D:/Integration-Game/gesture-trainer-web/python/check_video_duplicates.py)

You can use it in two ways:

1. scan a folder tree directly
2. scan a JSONL manifest with known `video_path`, `split`, and `label` fields

Example on a merged manifest:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python check_video_duplicates.py ^
  --manifest D:\Integration-Game\gesture-trainer-web\artifacts\word_model\everyday_daily_v1_merged\everyday_daily_v1_merged_manifest.jsonl ^
  --fail-on-cross-split
```

This is useful for:

- exact duplicate detection
- split-leakage detection between `train`, `val`, and `test`
- checking whether the same video appears under different labels

## Why this approach is safer than fully mixing datasets

It avoids several common problems:

- evaluation leakage across unrelated splits
- hidden label mismatches
- one dataset dominating the benchmark
- inflated metrics caused by inconsistent validation protocols

## What counts as success

The merge is worth keeping if it gives at least one of these:

- better validation accuracy
- better top-k behavior
- fewer confusion pairs on practical words
- more stable live predictions in the browser

## Notes

- `yt-dlp` is now part of the Python requirements for the automatic clip-preparation path
- the automatic path defaults to `train` clips only, which is the safest setup for augmentation
- if you already have local clips, you can reuse them with `--auto-ms-asl-skip-download`

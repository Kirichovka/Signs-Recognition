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

That strongly suggests the official package is **not a full self-contained video archive**. In practice, treat it as metadata, split files, or downloader assets until you inspect the extracted contents.

## Recommended merge strategy

Use this workflow:

1. Build the main subset from `ASL Citizen`
2. Build or prepare an `MS-ASL` manifest
3. Map `MS-ASL` labels to the same target labels as the ASL Citizen subset
4. Merge `MS-ASL` rows into the training split only
5. Keep evaluation focused on the base dataset

## Repository support

The repository now includes a merge helper:

- [`python/merge_sign_manifests.py`](/D:/Integration-Game/gesture-trainer-web/python/merge_sign_manifests.py)

It can:

- merge a base manifest with an extra manifest
- remap labels from the extra dataset
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

## Step 2. Prepare an MS-ASL manifest

This depends on the structure of the extracted `MS-ASL.zip` package.

At this stage, the safest workflow is:

1. download the official package
2. extract it
3. inspect the file tree
4. identify:
   - annotation files
   - split files
   - any provided video references or downloader assets

Until the exact package contents are confirmed on disk, do not assume the same structure as ASL Citizen.

## Step 3. Create a label map

The extra dataset must be mapped into the same label space as the base subset.

Examples of the kind of mapping you may need:

- `THANK YOU` -> `THANKYOU`
- `EAT` -> `EAT1`
- `DRINK` -> `DRINK1`
- `WELCOME` -> `WELCOME1`
- `WANT` -> `WANT1`

The merge helper expects a CSV with:

```text
source_label,target_label
THANK YOU,THANKYOU
EAT,EAT1
DRINK,DRINK1
WELCOME,WELCOME1
WANT,WANT1
```

## Step 4. Merge the manifests

Example command:

```bash
python merge_sign_manifests.py \
  --base-manifest ~/workspace/datasets/asl_citizen/asl_citizen_daily_v1_manifest.jsonl \
  --extra-manifest ~/workspace/datasets/ms_asl/ms_asl_daily_overlap_manifest.jsonl \
  --label-map ~/workspace/Signs-Recognition/python/label_maps/ms_asl_daily_map.csv \
  --allowed-labels-file ~/workspace/Signs-Recognition/python/label_sets/asl_citizen_daily_v1.txt \
  --force-extra-split train \
  --extra-source-name ms_asl \
  --base-source-name asl_citizen \
  --output ~/workspace/datasets/merged/everyday_daily_v1_merged_manifest.jsonl \
  --stats-output ~/workspace/datasets/merged/everyday_daily_v1_merged_stats.json
```

## Step 5. Train on the merged manifest

After merging:

1. run feature extraction on the merged manifest
2. train the GRU model exactly the same way as before
3. compare the result against the ASL Citizen-only baseline

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

## Recommended next step

Once `MS-ASL.zip` is downloaded and extracted, inspect the package and then:

1. identify its annotation structure
2. build a manifest
3. map overlapping everyday words
4. run the merge helper

At that point the repo is already ready for the merge step itself.

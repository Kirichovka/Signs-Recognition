# Alphabet Dataset

## Goal

This document describes the recommended image dataset path for an **ASL alphabet-only model**.

The key design decision is:

- keep the **word model** separate from the **alphabet model**

That separation matters because:

- the current word recognizer is a temporal model for signed words
- alphabet recognition from images is a mostly static classification task
- mixing both tasks into one first-pass model is likely to hurt quality

## Recommended dataset

Recommended static alphabet dataset:

- **ASL_SemCom**
- source: [Zenodo record](https://zenodo.org/records/14635573)

Why this dataset is a good first fit:

- direct downloadable archive
- RGB images rather than grayscale MNIST-style inputs
- explicit train/test image folders
- better aligned with a practical alphabet-only image model than a word-level video corpus

According to the dataset page:

- it contains static ASL alphabet hand images
- it excludes **J** and **Z** because those letters require motion
- each class contains `440` training images and `75` test images
- archive file: `ASL_SemCom.zip`
- published MD5: `831ff816c3bb36ffc3b0c9f248cf5033`

Source: [Zenodo dataset page](https://zenodo.org/records/14635573)

## Important limitation

This is a **static image dataset**.

That means:

- it is good for letters with stable handshapes
- it is **not** sufficient for motion letters like `J` and `Z`

Recommended handling:

- train the alphabet model on the static letters first
- add `J` and `Z` later as a small temporal extension if needed

## Download helper

The repository now includes:

- [`python/download_asl_semcom.py`](/D:/Integration-Game/gesture-trainer-web/python/download_asl_semcom.py)

It will:

- download the archive
- verify the MD5 checksum
- extract it into the local dataset folder

## Local download command

From the repository root:

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python download_asl_semcom.py
```

Default output path:

- [`datasets/asl_semcom`](/D:/Integration-Game/gesture-trainer-web/datasets/asl_semcom)

## Custom output directory

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python download_asl_semcom.py --output-dir D:\datasets\asl_semcom
```

## Download only, without extraction

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python download_asl_semcom.py --skip-extract
```

## Re-download even if the zip already exists

```powershell
cd D:\Integration-Game\gesture-trainer-web\python
python download_asl_semcom.py --force-download
```

## Expected structure after extraction

The exact folder names depend on the archive contents, but you should expect something like:

```text
datasets/
  asl_semcom/
    ASL_SemCom.zip
    ASL_SemCom/
      train/
      test/
```

## Why this is separate from the word model

Our current word recognizer:

- works on `40`-frame sequences
- uses hand + pose landmarks
- is optimized for isolated sign words

An alphabet image model should instead be treated as:

- image classification
- static handshape recognition
- a separate deployment mode

That gives a cleaner roadmap:

1. keep word recognition as a video/landmark model
2. build a separate alphabet recognizer from images
3. optionally combine both later at the UX layer, not in the first training step

## Suggested next step

After download, the next practical step is:

- inspect the folder structure
- confirm class names
- build a small alphabet training script or manifest for image classification

If needed, the repo can be extended next with:

- an alphabet dataset manifest builder
- an alphabet CNN training script
- a browser-friendly alphabet inference model

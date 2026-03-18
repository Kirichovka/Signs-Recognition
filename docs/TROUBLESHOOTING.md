# Troubleshooting

## 1. The camera does not start in the browser

### Symptoms

- `NotFoundError: Requested device not found`
- `Camera stream is not ready yet`
- the trainer stays in a waiting state

### What to check

1. Camera permission in the browser:
   - click the icon next to the address bar
   - set `Camera -> Allow`

2. Make sure the camera is not busy in another app:
   - Zoom
   - Discord
   - OBS
   - Teams
   - Windows Camera

3. Make sure you are opening the current page, not an old cached tab

4. Verify that the selected device supports the requested camera constraints

### What the code already does

The runtime now includes fallback attempts:

- explicit `deviceId`
- `facingMode: "user"`
- video without `facingMode`
- plain `video: true`

If the error still remains after that, the problem is probably outside the JS logic:

- browser permissions
- the camera device itself
- drivers
- another app using the camera

## 2. GitHub Pages opens, but `/api/health` returns 404

### Cause

You are opening an older backend-oriented flow.

### Fix

Use the current browser-only version.

The current architecture does not require:

- `/api/health`
- `/api/predict`

If you still see 404 on `/api/health`, it usually means:

- you opened an old URL
- or the browser loaded a cached build

Try:

- `Ctrl+F5`
- opening the page in a fresh tab
- verifying that the latest JS bundle is loaded

## 3. Server-side extraction fails with `libGL.so.1`

### Symptom

```text
ImportError: libGL.so.1: cannot open shared object file
```

### Fix

On Ubuntu:

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0
```

## 4. `unzip` is missing on the server

### Fix

```bash
sudo apt update
sudo apt install -y unzip
```

## 5. WLASL downloader shows many `Unsuccessful downloading` errors

### Cause

This is a normal problem with the older WLASL workflow:

- some links are outdated
- some videos depended on third-party hosts
- some entries point to `swf` or dead resources

### Practical fix

Use `ASL Citizen` as the primary dataset for fast and reliable training runs.

## 6. The browser model loads, but recognition quality feels weak

### Possible reasons

- the baseline model itself is still limited
- the current class set contains too many conflicting signs
- the user performs the sign in the wrong zone
- the camera does not capture hands, torso, or face clearly enough
- the model and metadata files do not match

### What to do

1. Verify that the `.onnx` file and metadata belong to the same training run
2. Check the live coaching and diagnostics on the page
3. Try a curated subset with more practical and visually distinct signs
4. Reduce the number of classes

## 7. Larger subset caps do not increase the number of videos

### Example

You set:

- `--max-train-per-class 120`
- `--max-val-per-class 30`

But you still get only around `30` videos for many classes.

### Cause

The dataset does not contain more examples for those signs.

The limits in `prepare_wlasl_subset.py`:

- cap class size from above
- but they do not create more data

## 8. Why letters are not included in the current everyday model

### Cause

- alphabet classes have too few examples
- some letters are missing or too weak as standalone classes
- mixing letters with everyday words would likely weaken the first model

### Recommended approach

Train separately:

- a words model
- an alphabet model

## 9. A Python backend is running locally, but GitHub Pages cannot use it

### Cause

GitHub Pages is static hosting only.

It does not:

- run FastAPI
- run PyTorch inference
- expose your local Python process as a built-in backend

### Fix

Either:

- use browser-only ONNX inference

Or:

- host a real backend separately

## 10. How to verify that the website is using the new model

Check:

- which label names appear in the UI
- which model and metadata files are referenced in the runtime
- whether any old API calls are still present
- whether the browser is serving a cached version

Useful files:

- [`js/sign-model-runtime.js`](/D:/Integration-Game/gesture-trainer-web/js/sign-model-runtime.js)
- [`models/asl_citizen_50.onnx`](/D:/Integration-Game/gesture-trainer-web/models/asl_citizen_50.onnx)
- [`models/asl_citizen_50_metadata.json`](/D:/Integration-Game/gesture-trainer-web/models/asl_citizen_50_metadata.json)


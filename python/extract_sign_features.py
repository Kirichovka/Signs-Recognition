from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


POSE_IDS = [0, 11, 12, 13, 14, 15, 16]
LEFT_HAND_SIZE = 21 * 3
RIGHT_HAND_SIZE = 21 * 3
POSE_SIZE = len(POSE_IDS) * 4
FEATURE_SIZE = LEFT_HAND_SIZE + RIGHT_HAND_SIZE + POSE_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract temporal landmark features from sign-language videos."
    )
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest path.")
    parser.add_argument("--output", required=True, help="Output NPZ path.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=48,
        help="Uniformly sampled frame count per video.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.3,
        help="Minimum pose visibility to trust shoulder normalization.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_frame_indices(total_frames: int, max_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=np.int32)
    if total_frames <= max_frames:
        return np.arange(total_frames, dtype=np.int32)
    return np.linspace(0, total_frames - 1, num=max_frames, dtype=np.int32)


def landmark_to_array(landmarks, count: int, with_visibility: bool) -> np.ndarray:
    if landmarks is None:
        width = 4 if with_visibility else 3
        return np.zeros((count, width), dtype=np.float32)
    rows = []
    for index in range(count):
        landmark = landmarks.landmark[index]
        if with_visibility:
            rows.append([landmark.x, landmark.y, landmark.z, getattr(landmark, "visibility", 0.0)])
        else:
            rows.append([landmark.x, landmark.y, landmark.z])
    return np.asarray(rows, dtype=np.float32)


def select_pose_subset(pose_landmarks) -> np.ndarray:
    if pose_landmarks is None:
        return np.zeros((len(POSE_IDS), 4), dtype=np.float32)
    rows = []
    for index in POSE_IDS:
        landmark = pose_landmarks.landmark[index]
        rows.append([landmark.x, landmark.y, landmark.z, getattr(landmark, "visibility", 0.0)])
    return np.asarray(rows, dtype=np.float32)


def normalize_landmarks(left_hand: np.ndarray, right_hand: np.ndarray, pose: np.ndarray, min_visibility: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_shoulder = pose[1]
    right_shoulder = pose[2]
    shoulder_visible = left_shoulder[3] >= min_visibility and right_shoulder[3] >= min_visibility

    if shoulder_visible:
        center_xy = (left_shoulder[:2] + right_shoulder[:2]) / 2.0
        scale = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
    else:
        wrist_candidates = []
        if np.any(left_hand):
            wrist_candidates.append(left_hand[0, :2])
        if np.any(right_hand):
            wrist_candidates.append(right_hand[0, :2])
        center_xy = np.mean(wrist_candidates, axis=0) if wrist_candidates else np.array([0.5, 0.5], dtype=np.float32)
        spread = []
        if np.any(left_hand):
            spread.append(np.linalg.norm(left_hand[0, :2] - left_hand[9, :2]))
        if np.any(right_hand):
            spread.append(np.linalg.norm(right_hand[0, :2] - right_hand[9, :2]))
        scale = float(np.mean(spread)) if spread else 0.15

    scale = max(scale, 1e-4)
    left_hand[:, :2] = (left_hand[:, :2] - center_xy) / scale
    right_hand[:, :2] = (right_hand[:, :2] - center_xy) / scale
    pose[:, :2] = (pose[:, :2] - center_xy) / scale
    left_hand[:, 2] /= scale
    right_hand[:, 2] /= scale
    pose[:, 2] /= scale
    return left_hand, right_hand, pose


def frame_feature_vector(results, min_visibility: float) -> np.ndarray:
    left_hand = landmark_to_array(results.left_hand_landmarks, 21, with_visibility=False)
    right_hand = landmark_to_array(results.right_hand_landmarks, 21, with_visibility=False)
    pose = select_pose_subset(results.pose_landmarks)
    left_hand, right_hand, pose = normalize_landmarks(left_hand, right_hand, pose, min_visibility)
    return np.concatenate([left_hand.reshape(-1), right_hand.reshape(-1), pose.reshape(-1)]).astype(np.float32)


def extract_video_features(video_path: Path, max_frames: int, min_visibility: float, holistic) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(sample_frame_indices(total_frames, max_frames).tolist())
    features = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            features.append(frame_feature_vector(results, min_visibility))
        frame_index += 1

    capture.release()

    if not features:
        return np.zeros((max_frames, FEATURE_SIZE), dtype=np.float32)

    array = np.asarray(features, dtype=np.float32)
    if len(array) < max_frames:
        padding = np.zeros((max_frames - len(array), FEATURE_SIZE), dtype=np.float32)
        array = np.concatenate([array, padding], axis=0)
    elif len(array) > max_frames:
        array = array[:max_frames]
    return array


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    records = load_manifest(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted({record["label"] for record in records})
    label_to_index = {label: index for index, label in enumerate(labels)}

    sequences = []
    label_ids = []
    splits = []
    signers = []
    video_ids = []

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for record in tqdm(records, desc="Extracting landmarks"):
            video_path = Path(record["video_path"])
            sequence = extract_video_features(video_path, args.max_frames, args.min_visibility, holistic)
            sequences.append(sequence)
            label_ids.append(label_to_index[record["label"]])
            splits.append(record.get("split", "train"))
            signers.append(-1 if record.get("signer_id") is None else int(record["signer_id"]))
            video_ids.append(record.get("video_id", video_path.stem))

    np.savez_compressed(
        output_path,
        sequences=np.asarray(sequences, dtype=np.float32),
        labels=np.asarray(label_ids, dtype=np.int64),
        splits=np.asarray(splits),
        signer_ids=np.asarray(signers, dtype=np.int64),
        video_ids=np.asarray(video_ids),
        label_names=np.asarray(labels),
        feature_size=np.asarray([FEATURE_SIZE], dtype=np.int32),
    )
    print(f"Saved features for {len(records)} videos to {output_path}")
    print(f"Classes: {len(labels)}")
    print(f"Feature size per frame: {FEATURE_SIZE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

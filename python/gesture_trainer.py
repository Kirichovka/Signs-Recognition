from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_LANDMARKS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


@dataclass
class GestureDefinition:
    gesture_id: str
    title: str
    instruction: str
    fingers: Dict[str, str]


@dataclass
class EvaluationResult:
    score: float
    matches: Dict[str, bool]
    expected: Dict[str, str]
    observed: Dict[str, str]


def load_config(config_path: Path) -> Tuple[float, float, List[GestureDefinition]]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    gestures = [
        GestureDefinition(
            gesture_id=item["id"],
            title=item["title"],
            instruction=item["instruction"],
            fingers=item["fingers"],
        )
        for item in data["gestures"]
    ]
    return data.get("hold_seconds", 1.0), data.get("score_threshold", 0.8), gestures


def angle_between(a, b, c) -> float:
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.hypot(*ab)
    mag_cb = math.hypot(*cb)
    if mag_ab == 0 or mag_cb == 0:
        return 180.0
    cosine = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    return math.degrees(math.acos(cosine))


def detect_finger_states(hand_landmarks, handedness_label: str) -> Dict[str, str]:
    lm = hand_landmarks.landmark
    states: Dict[str, str] = {}

    for finger_name, (mcp, pip, dip, tip) in FINGER_LANDMARKS.items():
        pip_angle = angle_between(lm[mcp], lm[pip], lm[dip])
        dip_angle = angle_between(lm[pip], lm[dip], lm[tip])
        fingertip_to_wrist = math.hypot(lm[tip].x - lm[0].x, lm[tip].y - lm[0].y)
        pip_to_wrist = math.hypot(lm[pip].x - lm[0].x, lm[pip].y - lm[0].y)

        if finger_name == "thumb":
            if handedness_label == "Right":
                stretched = lm[tip].x < lm[3].x and lm[tip].x < lm[mcp].x
            else:
                stretched = lm[tip].x > lm[3].x and lm[tip].x > lm[mcp].x
            states[finger_name] = "open" if stretched or pip_angle > 145 else "closed"
            continue

        straight = pip_angle > 160 and dip_angle > 150
        away_from_palm = fingertip_to_wrist > pip_to_wrist * 1.15
        states[finger_name] = "open" if straight and away_from_palm else "closed"

    return states


def evaluate_gesture(
    observed: Dict[str, str], definition: GestureDefinition
) -> EvaluationResult:
    expected = definition.fingers
    matches = {}
    total_checks = 0
    passed_checks = 0

    for finger_name in FINGER_NAMES:
        target = expected.get(finger_name, "any")
        actual = observed.get(finger_name, "unknown")
        matched = target == "any" or target == actual
        matches[finger_name] = matched
        if target != "any":
            total_checks += 1
            if matched:
                passed_checks += 1

    score = passed_checks / total_checks if total_checks else 0.0
    return EvaluationResult(
        score=score,
        matches=matches,
        expected=expected,
        observed=observed,
    )


def draw_panel(
    frame,
    gesture: GestureDefinition,
    result: EvaluationResult | None,
    hold_progress: float,
    stable_success: bool,
) -> None:
    cv2.rectangle(frame, (20, 20), (620, 230), (22, 26, 40), thickness=-1)
    cv2.putText(
        frame,
        f"Target: {gesture.title}",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        gesture.instruction,
        (40, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
    )
    cv2.putText(
        frame,
        "Keys: N-next  P-prev  R-reset hold  Q-quit",
        (40, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (170, 180, 190),
        1,
    )

    if result is None:
        cv2.putText(
            frame,
            "Show one hand to the camera.",
            (40, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 194, 255),
            2,
        )
        return

    score_text = f"Score: {int(result.score * 100)}%"
    score_color = (80, 220, 120) if result.score >= 0.8 else (0, 180, 255)
    cv2.putText(
        frame,
        score_text,
        (40, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        score_color,
        2,
    )
    cv2.putText(
        frame,
        f"Hold progress: {int(hold_progress * 100)}%",
        (220, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    parts = []
    for finger_name in FINGER_NAMES:
        expected = result.expected.get(finger_name, "any")
        observed = result.observed.get(finger_name, "unknown")
        mark = "OK" if result.matches.get(finger_name) else "X"
        parts.append(f"{finger_name}:{observed}/{expected} {mark}")
    cv2.putText(
        frame,
        " | ".join(parts),
        (40, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )

    if stable_success:
        cv2.putText(
            frame,
            "Great job! Gesture matched.",
            (40, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (70, 245, 110),
            2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype gesture trainer for simple child-friendly hand poses."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("gesture_config.json")),
        help="Path to the gesture configuration JSON file.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index passed to OpenCV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    hold_seconds, score_threshold, gestures = load_config(config_path)
    if not gestures:
        raise ValueError("The gesture configuration must define at least one gesture.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    gesture_index = 0
    hold_started_at: float | None = None
    last_success = False

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = hands.process(rgb_frame)
            current_gesture = gestures[gesture_index]
            result: EvaluationResult | None = None
            hold_progress = 0.0
            stable_success = False

            if output.multi_hand_landmarks and output.multi_handedness:
                hand_landmarks = output.multi_hand_landmarks[0]
                handedness = output.multi_handedness[0].classification[0].label
                observed = detect_finger_states(hand_landmarks, handedness)
                result = evaluate_gesture(observed, current_gesture)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

                if result.score >= score_threshold:
                    if hold_started_at is None:
                        hold_started_at = time.monotonic()
                    elapsed = time.monotonic() - hold_started_at
                    hold_progress = min(1.0, elapsed / hold_seconds)
                    stable_success = elapsed >= hold_seconds
                else:
                    hold_started_at = None
            else:
                hold_started_at = None

            if stable_success and not last_success:
                gesture_index = (gesture_index + 1) % len(gestures)
                hold_started_at = None
            last_success = stable_success

            draw_panel(frame, current_gesture, result, hold_progress, stable_success)
            cv2.imshow("Gesture Trainer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("n"):
                gesture_index = (gesture_index + 1) % len(gestures)
                hold_started_at = None
            if key == ord("p"):
                gesture_index = (gesture_index - 1) % len(gestures)
                hold_started_at = None
            if key == ord("r"):
                hold_started_at = None

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

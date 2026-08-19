"""MediaPipe hand-landmark extraction, isolated behind one small wrapper.

The rest of the codebase only ever sees:
  * a fixed-length float32 feature vector per frame (config.FEATURES), and
  * an annotated BGR frame with the hand skeleton drawn on it.

Hands are slotted deterministically by handedness (Left -> slot 0, Right -> slot 1)
so the feature vector layout is stable frame to frame, which the LSTM relies on.

MediaPipe 1.0 uses the Tasks API (HandLandmarker + a downloaded .task bundle).
All of that lives here; if the API ever changes only this file changes.
"""
from __future__ import annotations

import time
import urllib.request
from typing import Optional

import cv2
import numpy as np

import config

# Standard 21-point hand skeleton connections (thumb, index, middle, ring, pinky, palm).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                  # palm base
]

_HAND_COLORS = {0: (0, 200, 255), 1: (255, 120, 0)}  # slot 0 = Left, slot 1 = Right (BGR)


def _ensure_model() -> str:
    """Download the hand_landmarker.task bundle on first use; return its path."""
    path = config.HAND_MODEL_PATH
    if not path.exists():
        print(f"[landmarks] downloading hand model -> {path}")
        urllib.request.urlretrieve(config.HAND_MODEL_URL, path)
        print("[landmarks] download complete")
    return str(path)


class HandLandmarker:
    """Thin wrapper over MediaPipe Tasks HandLandmarker in VIDEO mode."""

    def __init__(self, num_hands: int = config.NUM_HANDS):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._mp_vision = mp_vision
        base = mp_python.BaseOptions(model_asset_path=_ensure_model())
        options = mp_vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._num_hands = num_hands
        self._t0 = time.monotonic()

    def _timestamp_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Run detection on one BGR frame.

        Returns (features, annotated_frame, hand_present):
          features        -> float32 vector of length config.FEATURES
          annotated_frame -> copy of the frame with the skeleton drawn
          hand_present    -> True if at least one hand was detected
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms())

        features = np.zeros(config.FEATURES, dtype=np.float32)
        annotated = frame_bgr.copy()
        hand_present = False

        if result.hand_landmarks:
            for hand_lms, handedness in zip(result.hand_landmarks, result.handedness):
                hand_present = True
                # slot 0 = Left, slot 1 = Right; falls back to slot 0 if single-hand config
                label = handedness[0].category_name if handedness else "Left"
                slot = 0 if label == "Left" else 1
                if slot >= self._num_hands:
                    slot = 0
                offset = slot * config.FEATURES_PER_HAND
                for i, lm in enumerate(hand_lms):
                    j = offset + i * config.COORDS
                    features[j] = lm.x
                    features[j + 1] = lm.y
                    features[j + 2] = lm.z
                _draw_hand(annotated, hand_lms, _HAND_COLORS.get(slot, (0, 255, 0)))

        return features, annotated, hand_present

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


def _draw_hand(frame_bgr: np.ndarray, hand_lms, color) -> None:
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame_bgr, p, 3, (255, 255, 255), -1)


def open_camera(index: int = config.CAMERA_INDEX) -> cv2.VideoCapture:
    """Open the webcam, preferring the DirectShow backend on Windows for speed."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


if __name__ == "__main__":
    # Phase 1 live overlay test: shows the skeleton on your hand. Press 'q' to quit.
    landmarker = HandLandmarker()
    cap = open_camera()
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (check CAMERA_INDEX in config.py)")
    print("Live overlay — press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror for a natural selfie view
            feats, annotated, present = landmarker.process(frame)
            status = "hand" if present else "no hand"
            cv2.putText(annotated, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("landmarks", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

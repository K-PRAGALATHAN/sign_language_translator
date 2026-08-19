"""Simple geometric gesture checks on raw hand landmarks.

Used for control gestures (not vocabulary signs). Currently: open-palm detection,
which the engine uses as the "end of sentence" trigger.

Works on a raw per-frame feature vector (config.FEATURES wide, MediaPipe normalized
image coords laid out as NUM_HANDS blocks of 21*(x,y,z)). It is orientation-tolerant
because "finger extended" is judged by distance-from-wrist, not by up/down position.
"""
from __future__ import annotations

import numpy as np

import config

# MediaPipe hand landmark indices
_WRIST = 0
_FINGER_TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky tips
_FINGER_PIPS = [6, 10, 14, 18]   # the joint one down from each tip
_THUMB_TIP = 4
_THUMB_MCP = 2


def _hand_points(features: np.ndarray, hand: int) -> np.ndarray | None:
    start = hand * config.FEATURES_PER_HAND
    block = features[start:start + config.FEATURES_PER_HAND]
    if not np.any(block):
        return None
    return block.reshape(config.LANDMARKS_PER_HAND, config.COORDS)[:, :2]  # x, y


def _is_open_palm_single(pts: np.ndarray, margin: float = 1.15) -> bool:
    wrist = pts[_WRIST]
    d = np.linalg.norm(pts - wrist, axis=1)
    # a finger is extended when its tip is meaningfully farther from the wrist
    # than the joint below it (fist -> tips curl in, closer than the pips)
    extended = sum(1 for tip, pip in zip(_FINGER_TIPS, _FINGER_PIPS)
                   if d[tip] > d[pip] * margin)
    thumb_out = d[_THUMB_TIP] > d[_THUMB_MCP]
    return extended >= 4 and thumb_out


def is_open_palm(features: np.ndarray) -> bool:
    """True if any detected hand is showing a clear, spread-open palm."""
    for hand in range(config.NUM_HANDS):
        pts = _hand_points(features, hand)
        if pts is not None and _is_open_palm_single(pts):
            return True
    return False

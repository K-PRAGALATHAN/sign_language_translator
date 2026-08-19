"""Landmark preprocessing shared by dataset building and real-time inference.

Pipeline per the spec:
  1. wrist-relative normalization  (removes absolute screen position)
  2. hand-size scaling             (removes distance-from-camera dependence)
  3. pad / truncate to a fixed length (uniform LSTM input shape)

A frame's feature vector is laid out as NUM_HANDS blocks of 63 = 21*(x,y,z).
Each hand block is normalized independently against its own wrist (landmark 0).
An all-zero block means "no hand in that slot" and is left as zeros.
"""
from __future__ import annotations

import numpy as np

import config

_WRIST = 0  # landmark index of the wrist within a hand


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize one frame (shape (FEATURES,)) hand-by-hand.

    For each present hand: subtract the wrist coordinate, then divide by the
    hand's overall scale (max distance of any landmark from the wrist).
    """
    out = frame.astype(np.float32).copy()
    for hand in range(config.NUM_HANDS):
        start = hand * config.FEATURES_PER_HAND
        block = out[start:start + config.FEATURES_PER_HAND]
        if not np.any(block):
            continue  # empty slot, keep zeros
        pts = block.reshape(config.LANDMARKS_PER_HAND, config.COORDS)
        wrist = pts[_WRIST].copy()
        pts -= wrist                                   # wrist-relative
        scale = np.linalg.norm(pts, axis=1).max()      # hand size
        if scale > 1e-6:
            pts /= scale                               # scale-normalize
        out[start:start + config.FEATURES_PER_HAND] = pts.reshape(-1)
    return out


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Normalize every frame of a (T, FEATURES) sequence."""
    return np.stack([normalize_frame(f) for f in seq]).astype(np.float32)


def pad_or_truncate(seq: np.ndarray, length: int = config.SEQ_LEN) -> np.ndarray:
    """Force a (T, FEATURES) sequence to exactly (length, FEATURES).

    Longer sequences are centre-cropped (keeps the middle of the gesture);
    shorter ones are zero-padded at the end.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != config.FEATURES:
        raise ValueError(f"expected (T, {config.FEATURES}), got {seq.shape}")
    t = seq.shape[0]
    if t == length:
        return seq
    if t > length:
        start = (t - length) // 2
        return seq[start:start + length]
    pad = np.zeros((length - t, config.FEATURES), dtype=np.float32)
    return np.concatenate([seq, pad], axis=0)


def preprocess_sequence(seq: np.ndarray, length: int = config.SEQ_LEN) -> np.ndarray:
    """Full pipeline: normalize each frame, then pad/truncate to `length`."""
    return pad_or_truncate(normalize_sequence(seq), length)

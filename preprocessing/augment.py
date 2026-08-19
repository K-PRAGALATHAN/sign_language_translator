"""Data augmentation for small per-class sample counts.

Operates on RAW sequences (T, FEATURES) before normalization:
  * horizontal mirroring  (flip x, and swap Left/Right hand slots)
  * time-warp             (resample the sequence to a slightly different speed)
  * coordinate jitter     (small Gaussian noise on all coordinates)

Each returns a new sequence; combine them in build_dataset.py.
"""
from __future__ import annotations

import numpy as np

import config

_rng = np.random.default_rng(config.RANDOM_SEED)


def mirror(seq: np.ndarray) -> np.ndarray:
    """Mirror horizontally: x -> 1-x for present hands, then swap hand slots.

    Landmarks are normalized image coords in [0,1], so a horizontal flip maps
    x to 1-x. A left hand becomes a right hand, so we also swap the two 63-wide
    slots to keep the Left-slot-0 / Right-slot-1 convention consistent.
    """
    out = seq.copy()
    for hand in range(config.NUM_HANDS):
        start = hand * config.FEATURES_PER_HAND
        block = out[:, start:start + config.FEATURES_PER_HAND]
        if not np.any(block):
            continue
        pts = block.reshape(seq.shape[0], config.LANDMARKS_PER_HAND, config.COORDS)
        pts[:, :, 0] = 1.0 - pts[:, :, 0]  # flip x
        out[:, start:start + config.FEATURES_PER_HAND] = pts.reshape(seq.shape[0], -1)

    if config.NUM_HANDS == 2:
        a = out[:, :config.FEATURES_PER_HAND].copy()
        b = out[:, config.FEATURES_PER_HAND:2 * config.FEATURES_PER_HAND].copy()
        out[:, :config.FEATURES_PER_HAND] = b
        out[:, config.FEATURES_PER_HAND:2 * config.FEATURES_PER_HAND] = a
    return out


def time_warp(seq: np.ndarray, factor: float | None = None) -> np.ndarray:
    """Resample the sequence along time to simulate faster/slower signing."""
    t = seq.shape[0]
    if factor is None:
        factor = float(_rng.uniform(0.8, 1.2))
    new_t = max(2, int(round(t * factor)))
    src = np.linspace(0, t - 1, num=new_t)
    idx = np.clip(np.round(src).astype(int), 0, t - 1)
    return seq[idx]


def jitter(seq: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Add small Gaussian noise, but only to non-empty (hand-present) frames/slots."""
    out = seq.copy()
    for hand in range(config.NUM_HANDS):
        start = hand * config.FEATURES_PER_HAND
        block = out[:, start:start + config.FEATURES_PER_HAND]
        mask = np.any(block != 0, axis=1)  # frames where this hand is present
        noise = _rng.normal(0.0, sigma, size=block.shape).astype(np.float32)
        block[mask] += noise[mask]
    return out


def augmentations(seq: np.ndarray) -> list[np.ndarray]:
    """Return a list of augmented variants (not including the original)."""
    return [
        mirror(seq),
        time_warp(seq),
        jitter(seq),
        jitter(time_warp(seq)),
    ]

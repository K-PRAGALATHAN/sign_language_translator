"""Turn recorded raw clips into a training-ready dataset.

Reads data/raw/<WORD>/*.npy, normalizes + pads each sequence to (SEQ_LEN, FEATURES),
optionally adds augmented variants, and writes:
  data/processed/X.npy          float32 (N, SEQ_LEN, FEATURES)
  data/processed/y.npy          int64   (N,)  class indices
  data/processed/label_map.json {"idx_to_word": [...], "word_to_idx": {...}}

Run from the project root:  python -m preprocessing.build_dataset
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import config
from common import preprocess as P
from preprocessing import augment as A


def discover_words() -> list[str]:
    """Words that actually have recorded clips, ordered by config.VOCAB first."""
    present = {d.name for d in config.RAW_DIR.iterdir() if d.is_dir()
               and any(d.glob("*.npy"))}
    ordered = [w for w in config.VOCAB if w in present]
    extra = sorted(present - set(config.VOCAB))
    return ordered + extra


def build(augment: bool) -> None:
    words = discover_words()
    if not words:
        raise SystemExit(
            f"No recorded clips found under {config.RAW_DIR}. "
            "Run `python -m collect.collect_data` first."
        )

    word_to_idx = {w: i for i, w in enumerate(words)}
    X: list[np.ndarray] = []
    y: list[int] = []
    per_word_raw: dict[str, int] = {}

    for word in words:
        clips = sorted((config.RAW_DIR / word).glob("*.npy"))
        per_word_raw[word] = len(clips)
        for clip_path in clips:
            raw = np.load(clip_path)                    # (T, FEATURES)
            variants = [raw]
            if augment:
                variants += A.augmentations(raw)
            for seq in variants:
                X.append(P.preprocess_sequence(seq))    # normalize + pad
                y.append(word_to_idx[word])

    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int64)

    np.save(config.X_PATH, X_arr)
    np.save(config.Y_PATH, y_arr)
    label_map = {"idx_to_word": words, "word_to_idx": word_to_idx}
    config.PROCESSED_LABEL_MAP_PATH.write_text(json.dumps(label_map, indent=2))

    print(f"words ({len(words)}): {', '.join(words)}")
    print("raw clips per word:")
    for w in words:
        print(f"  {config.display_word(w):12s} {per_word_raw[w]}")
    print(f"augmentation: {'on' if augment else 'off'}")
    print(f"X {X_arr.shape}  y {y_arr.shape}")
    print(f"saved -> {config.X_PATH}")
    print(f"saved -> {config.Y_PATH}")
    print(f"saved -> {config.PROCESSED_LABEL_MAP_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build processed dataset from raw clips")
    ap.add_argument("--no-augment", action="store_true", help="disable augmentation")
    args = ap.parse_args()
    build(augment=config.USE_AUGMENTATION and not args.no_augment)


if __name__ == "__main__":
    main()

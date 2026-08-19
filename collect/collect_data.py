"""Guided webcam recorder for building the training set.

For each word in the vocabulary you record several short clips. Each clip is a
sequence of raw per-frame landmark vectors (config.FEATURES wide), saved as a
.npy under data/raw/<WORD>/<n>.npy. Preprocessing/normalization happens later in
build_dataset.py so the raw captures stay reusable.

Controls (during the window):
  SPACE  start recording the next clip for the current word (after a countdown)
  n      skip to the next word
  b      go back to the previous word
  d      delete the most recent clip for the current word
  q      quit

Run from the project root:  python -m collect.collect_data
Optional: python -m collect.collect_data --words ME WANT WATER --samples 40
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import config
from common.landmarks import HandLandmarker, open_camera


def _existing_count(word: str) -> int:
    d = config.RAW_DIR / word
    if not d.exists():
        return 0
    return len(list(d.glob("*.npy")))


def _save_clip(word: str, frames: list[np.ndarray]) -> int:
    d = config.RAW_DIR / word
    d.mkdir(parents=True, exist_ok=True)
    idx = _existing_count(word)
    np.save(d / f"{idx:03d}.npy", np.stack(frames).astype(np.float32))
    return idx


def _delete_last(word: str) -> str | None:
    d = config.RAW_DIR / word
    files = sorted(d.glob("*.npy")) if d.exists() else []
    if not files:
        return None
    last = files[-1]
    last.unlink()
    return last.name


def _banner(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 30
    for text, color in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 30


def run(words: list[str], samples_per_word: int, clip_len: int) -> None:
    landmarker = HandLandmarker()
    cap = open_camera()
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (check CAMERA_INDEX in config.py)")

    wi = 0
    recording = False
    countdown_until = 0.0
    clip_frames: list[np.ndarray] = []

    print("SPACE=record  n=next word  b=prev word  d=delete last  q=quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            feats, annotated, present = landmarker.process(frame)

            word = words[wi]
            have = _existing_count(word)
            now = time.monotonic()

            if countdown_until and now < countdown_until:
                remaining = int(countdown_until - now) + 1
                cv2.putText(annotated, str(remaining),
                            (annotated.shape[1] // 2 - 20, annotated.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 5)
            elif countdown_until and now >= countdown_until:
                countdown_until = 0.0
                recording = True
                clip_frames = []

            if recording:
                clip_frames.append(feats)
                cv2.circle(annotated, (annotated.shape[1] - 30, 30), 12, (0, 0, 255), -1)
                if len(clip_frames) >= clip_len:
                    idx = _save_clip(word, clip_frames)
                    have = _existing_count(word)
                    recording = False
                    print(f"saved {word}/{idx:03d}.npy  ({have}/{samples_per_word})")

            rec_state = "RECORDING" if recording else ("READY" if not countdown_until else "GET READY")
            _banner(annotated, [
                (f"WORD: {config.display_word(word)}  [{wi + 1}/{len(words)}]", (0, 255, 255)),
                (f"clips: {have}/{samples_per_word}   {rec_state}", (0, 255, 0)),
                ("SPACE record | n next | b prev | d delete | q quit", (200, 200, 200)),
            ])
            cv2.imshow("collect", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if recording or countdown_until:
                continue  # ignore navigation while busy
            if key == ord(" "):
                countdown_until = now + config.COUNTDOWN_SECONDS
            elif key == ord("n"):
                wi = (wi + 1) % len(words)
            elif key == ord("b"):
                wi = (wi - 1) % len(words)
            elif key == ord("d"):
                removed = _delete_last(word)
                print(f"deleted {word}/{removed}" if removed else f"no clips for {word}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    print("\nClip counts:")
    for w in words:
        print(f"  {config.display_word(w):12s} {_existing_count(w)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Webcam sign data collector")
    ap.add_argument("--words", nargs="*", default=config.VOCAB,
                    help="words to record (default: full config.VOCAB)")
    ap.add_argument("--samples", type=int, default=config.SAMPLES_PER_WORD,
                    help="target clips per word (display only)")
    ap.add_argument("--clip-len", type=int, default=config.SEQ_LEN,
                    help="frames captured per clip")
    args = ap.parse_args()
    run([w.upper() for w in args.words], args.samples, args.clip_len)


if __name__ == "__main__":
    main()

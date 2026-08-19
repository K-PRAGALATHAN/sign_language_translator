"""Standalone end-to-end demo in an OpenCV window (no web server needed).

Webcam -> MediaPipe landmarks -> LSTM -> gloss builder -> (on sentence finalize)
OpenRouter gloss->sentence -> spoken aloud + printed. This is the quickest way to
prove the whole laptop pipeline works.

Controls:
  SPACE  finalize the current sentence now (stands in for the hardware button)
  c      clear the current gloss
  q      quit

Run from the project root:  python -m inference.realtime_demo
"""
from __future__ import annotations

import cv2
import numpy as np

import config
from common.landmarks import HandLandmarker, open_camera
from inference.engine import RecognitionEngine
from inference.gloss_to_sentence import gloss_to_sentence
from inference import tts
from model.model import load_model


def _handle_sentence(gloss: list[str], history: list[str]) -> None:
    sentence, source = gloss_to_sentence(gloss)
    line = f"{' '.join(config.display_word(w) for w in gloss)}  ->  {sentence}  [{source}]"
    print(line)
    history.append(sentence)
    tts.speak(sentence)


def main() -> None:
    if not config.MODEL_PATH.exists():
        raise SystemExit(
            "No trained model found. Record data, build the dataset, and train first:\n"
            "  python -m collect.collect_data\n"
            "  python -m preprocessing.build_dataset\n"
            "  python -m model.train"
        )
    model, words = load_model()
    engine = RecognitionEngine(model, words)
    landmarker = HandLandmarker()
    cap = open_camera()
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (check CAMERA_INDEX in config.py)")

    history: list[str] = []
    print("Signing... SPACE=finalize sentence  c=clear  q=quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            feats, annotated, present = landmarker.process(frame)
            event = engine.update(feats, present)

            if event.sentence_gloss:
                _handle_sentence(event.sentence_gloss, history)

            # overlay: current guess, running gloss, last sentence
            if event.top_word:
                cv2.putText(annotated, f"{config.display_word(event.top_word)} {event.top_conf:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            gloss_txt = " ".join(config.display_word(w) for w in event.gloss)
            cv2.putText(annotated, f"GLOSS: {gloss_txt}", (10, annotated.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if history:
                cv2.putText(annotated, history[-1][:60], (10, annotated.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("realtime demo", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                ev = engine.finalize()
                if ev.sentence_gloss:
                    _handle_sentence(ev.sentence_gloss, history)
            elif key == ord("c"):
                engine.reset_sentence()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()

"""Real-time recognition engine: frames in, confirmed glosses and sentences out.

Drives the whole recognition side of the pipeline so both the OpenCV demo and the
FastAPI dashboard can share identical behaviour. It is camera- and UI-agnostic:
you feed it raw per-frame feature vectors, it returns lightweight events.

Logic (per the spec section 6):
  * rolling buffer of the last SEQ_LEN frames
  * run the model every INFERENCE_STRIDE frames (sliding window)
  * a prediction must stay above CONFIDENCE_THRESHOLD for CONSECUTIVE_WINDOWS
    windows to be *confirmed*
  * debounce: don't append the same sign twice in a row while the hand holds
  * append confirmed signs to the running gloss list
  * if no new sign for SENTENCE_TIMEOUT_S (or finalize() is called), the sentence
    is complete -> emitted for gloss->sentence + TTS
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config
from common import preprocess as P
from common.gestures import is_open_palm
from model.model import SignLSTM, predict


@dataclass
class EngineEvent:
    """What happened on one update() call. Any field may be None/empty."""
    top_word: str | None = None          # current best guess (for live display)
    top_conf: float = 0.0
    confirmed_sign: str | None = None     # a sign just got appended to the gloss
    gloss: list[str] = field(default_factory=list)  # current running gloss
    sentence_gloss: list[str] | None = None  # non-None when a sentence just finalized
    palm_active: bool = False             # open palm currently being held (for UI feedback)
    palm_progress: float = 0.0            # 0..1 toward the required hold time


class RecognitionEngine:
    def __init__(self, model: SignLSTM, words: list[str], device: str = "cpu"):
        self.model = model
        self.words = words
        self.device = device
        self.buffer: deque[np.ndarray] = deque(maxlen=config.SEQ_LEN)
        self.frame_count = 0
        self.recent_preds: deque[int] = deque(maxlen=config.CONSECUTIVE_WINDOWS)
        self.gloss: list[str] = []
        self.last_confirmed: str | None = None
        self.last_activity = time.monotonic()
        self.hand_frames = 0
        self.palm_start: float | None = None

    def reset_sentence(self) -> None:
        self.gloss = []
        self.last_confirmed = None
        self.recent_preds.clear()
        self.buffer.clear()
        self.hand_frames = 0
        self.palm_start = None

    def update(self, features: np.ndarray, hand_present: bool,
               now: float | None = None) -> EngineEvent:
        now = time.monotonic() if now is None else now
        self.buffer.append(features.astype(np.float32))
        if hand_present:
            self.hand_frames = min(self.hand_frames + 1, config.SEQ_LEN)
        else:
            self.hand_frames = max(self.hand_frames - 1, 0)
        self.frame_count += 1

        event = EngineEvent(gloss=list(self.gloss))

        # 1) open-palm "end of sentence" gesture, held for PALM_HOLD_SECONDS (real
        #    time, so it's independent of frame rate). While a palm is held we
        #    neither run the classifier nor let those frames confirm a sign, so the
        #    palm itself is never mistaken for a vocabulary word. Only when the hold
        #    completes is the gloss finalized and sent onward to the LLM.
        if hand_present and is_open_palm(features):
            if self.palm_start is None:
                self.palm_start = now
            self.recent_preds.clear()
            held = now - self.palm_start
            event.palm_active = True
            event.palm_progress = min(held / config.PALM_HOLD_SECONDS, 1.0)
            if held >= config.PALM_HOLD_SECONDS and self.gloss:
                event.sentence_gloss = list(self.gloss)
                self.reset_sentence()
                event.gloss = []
            return event
        self.palm_start = None

        # 2) optional inactivity timeout (off in palm-only mode; see config.USE_TIMEOUT)
        if (config.USE_TIMEOUT and self.gloss
                and (now - self.last_activity) >= config.SENTENCE_TIMEOUT_S):
            event.sentence_gloss = list(self.gloss)
            self.reset_sentence()
            event.gloss = []
            return event

        # 3) only run the model on a full buffer that actually contains a hand
        run = (len(self.buffer) == config.SEQ_LEN
               and self.frame_count % config.INFERENCE_STRIDE == 0
               and self.hand_frames >= config.MIN_HAND_FRAMES)
        if not run:
            return event

        seq = P.preprocess_sequence(np.stack(self.buffer))
        idx, conf = predict(self.model, seq, self.device)
        event.top_word = self.words[idx]
        event.top_conf = conf

        # 4) confidence gate + persistence across consecutive windows
        if conf >= config.CONFIDENCE_THRESHOLD:
            self.recent_preds.append(idx)
        else:
            self.recent_preds.clear()

        stable = (len(self.recent_preds) == config.CONSECUTIVE_WINDOWS
                  and len(set(self.recent_preds)) == 1)
        if stable:
            word = self.words[self.recent_preds[0]]
            # 5) debounce duplicates while the hand holds the same sign
            if word != self.last_confirmed:
                self.gloss.append(word)
                self.last_confirmed = word
                self.last_activity = now
                self.recent_preds.clear()
                event.confirmed_sign = word
                event.gloss = list(self.gloss)
        return event

    def finalize(self) -> EngineEvent:
        """Force sentence completion (the 'button press' / dashboard button)."""
        event = EngineEvent(gloss=list(self.gloss))
        if self.gloss:
            event.sentence_gloss = list(self.gloss)
            self.reset_sentence()
            event.gloss = []
        return event

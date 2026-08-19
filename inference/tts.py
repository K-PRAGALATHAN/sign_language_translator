"""Text-to-speech via pyttsx3, one utterance per subprocess.

Why a subprocess: pyttsx3.init() returns a cached singleton engine, and after the
first runAndWait() its internal run-loop is finished — so reusing it in-process makes
only the FIRST utterance audible and silently drops the rest. Running each utterance
in a fresh Python process sidesteps that entirely and is rock-solid on Windows SAPI5.

Utterances are queued on a single background worker thread so speaking never blocks
the recognition loop, and sentences don't overlap.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading

# Child program: read the text from stdin, speak it, exit. Reading from stdin
# avoids any argv quoting/escaping issues with punctuation in the sentence.
_CHILD = (
    "import sys, pyttsx3;"
    "t = sys.stdin.read();"
    "e = pyttsx3.init();"
    "e.say(t);"
    "e.runAndWait();"
    "e.stop()"
)


def _speak_blocking(text: str) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        proc.communicate(text.encode("utf-8"), timeout=30)
    except Exception:
        proc.kill()


class Speaker:
    def __init__(self) -> None:
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        if text and text.strip():
            self._q.put(text)

    def _worker(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                return
            try:
                _speak_blocking(text)
            except Exception as e:
                print(f"[tts] speak failed: {e}")

    def close(self) -> None:
        self._q.put(None)


# Simple module-level singleton for convenience.
_default: Speaker | None = None


def speak(text: str) -> None:
    global _default
    if _default is None:
        _default = Speaker()
    _default.say(text)


if __name__ == "__main__":
    import time
    # Speak three sentences in a row — all three should be audible.
    for s in ["First sentence.", "Second sentence.", "Third sentence."]:
        speak(s)
    time.sleep(12)

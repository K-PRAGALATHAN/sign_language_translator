"""Send a sentence (text + TTS audio) to the ESP8266 output node over WiFi.

The node (firmware/OutputNode) shows the text on its OLED and plays the audio on
its MAX98357A speaker. Protocol (raw TCP): text line, sample-rate line, then raw
16-bit signed little-endian mono PCM until the socket is closed.

pyttsx3 on Windows already emits mono 16-bit PCM (22050 Hz), so no resampling is
needed; for any other format we down-mix / re-quantize with NumPy.

Usage:
    python -m inference.output_node "I want water."          # uses config.OUTPUT_NODE_IP
    python -m inference.output_node "Hello" 172.20.10.2
"""
from __future__ import annotations

import os
import socket
import tempfile
import wave

import config


def synthesize(text: str) -> tuple[int, bytes]:
    """Render `text` to (sample_rate, mono-16bit-PCM-bytes) via pyttsx3."""
    import pyttsx3

    path = os.path.join(tempfile.gettempdir(), "slt_tts_node.wav")
    engine = pyttsx3.init()
    engine.save_to_file(text, path)
    engine.runAndWait()
    engine.stop()

    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        frames = w.readframes(w.getnframes())

    if ch == 1 and sw == 2:
        return rate, frames  # already the format the node wants

    import numpy as np
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if sw == 1:
        a = (a - 128.0) / 128.0
    else:
        a = a / float(1 << (8 * sw - 1))
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    pcm = np.clip(a * 32767.0, -32768, 32767).astype("<i2").tobytes()
    return rate, pcm


def send(text: str, host: str | None = None, port: int | None = None,
         timeout: float = 10.0) -> bool:
    """Speak+show `text` on the output node. Returns True on success."""
    host = host or config.OUTPUT_NODE_IP
    port = port or config.OUTPUT_NODE_PORT
    if not text or not text.strip():
        return False
    rate, pcm = synthesize(text)
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall((text.replace("\n", " ").strip() + "\n").encode("utf-8"))
            s.sendall((str(rate) + "\n").encode("ascii"))
            s.sendall(pcm)
            s.shutdown(socket.SHUT_WR)   # EOF -> node knows the audio ended
        return True
    except OSError as e:
        print(f"[output_node] send failed ({host}:{port}): {e}")
        return False


if __name__ == "__main__":
    import sys
    msg = sys.argv[1] if len(sys.argv) > 1 else "I want some water."
    host = sys.argv[2] if len(sys.argv) > 2 else None
    ok = send(msg, host=host)
    print("sent" if ok else "failed", "->", host or config.OUTPUT_NODE_IP)

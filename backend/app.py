"""FastAPI backend serving the live dashboard.

One background thread owns the webcam + MediaPipe + RecognitionEngine. It:
  * publishes the latest annotated frame as an MJPEG stream (GET /video)
  * pushes recognition events (top guess, running gloss, finalized sentence with
    the spoken text) to all connected dashboards over a WebSocket (/ws)

Endpoints:
  GET  /            -> dashboard/index.html
  GET  /video       -> multipart MJPEG stream of the annotated feed
  WS   /ws          -> JSON events
  POST /finalize    -> force sentence completion (dashboard button / hardware button)

Run from the project root:  python -m backend.app
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

import config
from common.landmarks import HandLandmarker, open_camera
from inference.engine import RecognitionEngine
from inference.gloss_to_sentence import gloss_to_sentence
from inference import tts, output_node
from model.model import load_model

app = FastAPI(title="Sign Language Translator")


class CameraWorker:
    """Owns the camera loop; exposes the latest JPEG and an event queue."""

    def __init__(self) -> None:
        self.latest_jpeg: bytes | None = None
        self.events: list[dict] = []          # drained by the WS broadcaster
        self._lock = threading.Lock()
        self._finalize_flag = threading.Event()
        self._stop = threading.Event()
        self._history: list[dict] = []
        self.model = None
        self.words: list[str] = []
        self.ready = False
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def request_finalize(self) -> None:
        self._finalize_flag.set()

    def _emit(self, event: dict) -> None:
        with self._lock:
            self.events.append(event)

    def drain_events(self) -> list[dict]:
        with self._lock:
            out, self.events = self.events, []
        return out

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self.latest_jpeg

    def _handle_sentence(self, gloss: list[str]) -> None:
        # Do the LLM call + output on a background thread so the camera loop
        # never stalls (OpenRouter + node streaming can take a couple seconds).
        threading.Thread(target=self._process_sentence,
                         args=(list(gloss),), daemon=True).start()

    def _process_sentence(self, gloss: list[str]) -> None:
        sentence, source = gloss_to_sentence(gloss)
        entry = {
            "type": "sentence",
            "gloss": [config.display_word(w) for w in gloss],
            "sentence": sentence,
            "source": source,
            "ts": time.strftime("%H:%M:%S"),
        }
        self._history.append(entry)
        self._emit(entry)
        # Speak/show on the ESP output node (OLED + speaker). Falls back to the
        # laptop speaker only if the node is disabled.
        if config.OUTPUT_NODE_ENABLED:
            output_node.send(sentence)
        else:
            tts.speak(sentence)

    def _run(self) -> None:
        try:
            if not config.MODEL_PATH.exists():
                raise RuntimeError("No trained model — run collect/build/train first.")
            self.model, self.words = load_model()
            engine = RecognitionEngine(self.model, self.words)
            landmarker = HandLandmarker()
            cap = open_camera()
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam (check CAMERA_INDEX).")
            self.ready = True
        except Exception as e:
            self.error = str(e)
            self._emit({"type": "error", "message": str(e)})
            return

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            feats, annotated, present = landmarker.process(frame)
            event = engine.update(feats, present)

            if self._finalize_flag.is_set():
                self._finalize_flag.clear()
                ev = engine.finalize()
                if ev.sentence_gloss:
                    self._handle_sentence(ev.sentence_gloss)

            if event.sentence_gloss:
                self._handle_sentence(event.sentence_gloss)
            else:
                self._emit({
                    "type": "state",
                    "top_word": (config.display_word(event.top_word)
                                 if event.top_word else None),
                    "top_conf": round(event.top_conf, 3),
                    "gloss": [config.display_word(w) for w in event.gloss],
                    "confirmed": (config.display_word(event.confirmed_sign)
                                  if event.confirmed_sign else None),
                    "palm_active": event.palm_active,
                    "palm_progress": round(event.palm_progress, 2),
                })

            ok2, buf = cv2.imencode(".jpg", annotated,
                                    [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            if ok2:
                with self._lock:
                    self.latest_jpeg = buf.tobytes()

        cap.release()
        landmarker.close()


worker = CameraWorker()


@app.on_event("startup")
def _startup() -> None:
    worker.start()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.ROOT / "dashboard" / "index.html")


def _mjpeg():
    boundary = b"--frame"
    while True:
        jpeg = worker.get_jpeg()
        if jpeg is None:
            time.sleep(0.03)
            continue
        yield (boundary + b"\r\nContent-Type: image/jpeg\r\n"
               + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
        time.sleep(0.02)


@app.get("/video")
def video() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/finalize")
def finalize() -> dict:
    worker.request_finalize()
    return {"ok": True}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    # send any history so a late-joining dashboard has context
    try:
        await websocket.send_text(json.dumps(
            {"type": "hello", "vocab": [config.display_word(w) for w in config.VOCAB],
             "palm_hold_seconds": config.PALM_HOLD_SECONDS,
             "history": worker._history}))
        while True:
            for event in worker.drain_events():
                await websocket.send_text(json.dumps(event))
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return
    except Exception:
        return


def main() -> None:
    import uvicorn
    print(f"Dashboard -> http://{config.HOST}:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()

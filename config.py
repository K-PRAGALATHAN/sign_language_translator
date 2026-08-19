"""Single source of truth for all pipeline knobs.

Every module imports from here so behaviour stays consistent across data
collection, training, and real-time inference. Adjust values here rather than
scattering constants through the codebase.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no external dependency).

    Reads KEY=VALUE lines from `path` into os.environ WITHOUT overriding values
    already set in the real environment (a shell export always wins). Supports
    `#` comments, blank lines, optional surrounding quotes, and an optional
    leading `export `.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # data/raw/<WORD>/<n>.npy
PROCESSED_DIR = DATA_DIR / "processed"     # X.npy, y.npy, label_map.json
MODEL_DIR = ROOT / "model" / "saved_model"
MEDIAPIPE_DIR = ROOT / "models"            # downloaded hand_landmarker.task lives here

for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, MEDIAPIPE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "sign_lstm.pt"
LABEL_MAP_PATH = MODEL_DIR / "label_map.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

X_PATH = PROCESSED_DIR / "X.npy"
Y_PATH = PROCESSED_DIR / "y.npy"
PROCESSED_LABEL_MAP_PATH = PROCESSED_DIR / "label_map.json"

# --------------------------------------------------------------------------- #
# Vocabulary  (start small for the MVP; expand later)
# --------------------------------------------------------------------------- #
VOCAB = [
    "HELLO",
    "ME",
    "WANT",
    "WATER",
    "HELP",
    "THANK_YOU",
]

# --------------------------------------------------------------------------- #
# Landmark / sequence shape
# --------------------------------------------------------------------------- #
NUM_HANDS = 2               # 2 -> 126 features/frame; set 1 -> 63 (spec-exact)
LANDMARKS_PER_HAND = 21
COORDS = 3                  # x, y, z
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS          # 63
FEATURES = NUM_HANDS * FEATURES_PER_HAND                 # 126 (or 63)

SEQ_LEN = 30                # frames per sample (pad/truncate to this)

# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #
SAMPLES_PER_WORD = 30       # target recorded clips per word
COUNTDOWN_SECONDS = 2       # get-ready countdown before each clip
CAMERA_INDEX = 0

# Camera source: a webcam index (0, 1, ...) OR an ESP32-CAM MJPEG stream URL,
# e.g. "http://172.20.10.2:81/stream". Override with the CAMERA_SOURCE env var.
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", str(CAMERA_INDEX))

# --------------------------------------------------------------------------- #
# MediaPipe
# --------------------------------------------------------------------------- #
HAND_MODEL_PATH = MEDIAPIPE_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
LSTM1_UNITS = 64
LSTM2_UNITS = 32
DENSE_UNITS = 32
DROPOUT = 0.3
EPOCHS = 120
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15            # test is the remaining 0.15
RANDOM_SEED = 42
USE_AUGMENTATION = True

# --------------------------------------------------------------------------- #
# Real-time inference
# --------------------------------------------------------------------------- #
CONFIDENCE_THRESHOLD = 0.85     # below this -> "no sign"
CONSECUTIVE_WINDOWS = 5         # windows a sign must persist to be confirmed
INFERENCE_STRIDE = 3            # run model every N frames
MIN_HAND_FRAMES = 5             # min frames with a hand present in a window to bother predicting

# Sentence completion. Palm-only mode: a sentence ends when you hold an open palm
# to the camera for PALM_HOLD_SECONDS (or press the finalize button). Only then is
# the gloss sent to the LLM. The auto-timeout is off by default so it never cuts you
# off mid-sentence; flip USE_TIMEOUT = True to re-enable it.
USE_TIMEOUT = False
SENTENCE_TIMEOUT_S = 2.5        # only used when USE_TIMEOUT is True
PALM_HOLD_SECONDS = 3.0         # seconds an open palm must be held to end the sentence

# --------------------------------------------------------------------------- #
# Gloss -> sentence (OpenRouter, OpenAI-compatible)
# --------------------------------------------------------------------------- #
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "anthropic/claude-haiku-4.5"
)
OPENROUTER_TIMEOUT_S = 15

# --------------------------------------------------------------------------- #
# Backend / dashboard
# --------------------------------------------------------------------------- #
HOST = "127.0.0.1"
PORT = 8000
JPEG_QUALITY = 70

# --------------------------------------------------------------------------- #
# ESP8266 output node (OLED + speaker over WiFi)
# --------------------------------------------------------------------------- #
OUTPUT_NODE_ENABLED = os.environ.get("OUTPUT_NODE_ENABLED", "1") == "1"
OUTPUT_NODE_IP = os.environ.get("OUTPUT_NODE_IP", "172.20.10.3")
OUTPUT_NODE_PORT = int(os.environ.get("OUTPUT_NODE_PORT", "9001"))


def display_word(word: str) -> str:
    """Human-readable form of a vocabulary token (THANK_YOU -> 'THANK YOU')."""
    return word.replace("_", " ")

# Real-Time Sign Language → Speech Translator (laptop build)

Webcam → MediaPipe hand landmarks → LSTM sign classifier → gloss builder →
LLM gloss-to-sentence → spoken audio + live dashboard.

This is the **laptop-first** implementation of
[`Sign_Language_Translator_Full_Spec.md`](Sign_Language_Translator_Full_Spec.md).
The ESP32-CAM, OLED, speaker-amp, and physical button are deferred to the hardware
phase — on the laptop, the webcam, browser dashboard, laptop speaker, and a keyboard/
dashboard button stand in for them.

## What differs from the spec (intentional)

| Spec | Here | Why |
|---|---|---|
| TensorFlow/Keras | **PyTorch 2.13** | No TensorFlow wheels for Python 3.14 (spec allows PyTorch) |
| WLASL dataset | **Self-recorded webcam clips** | No dead YouTube links; training matches real inference conditions |
| 63 features (1 hand) | **126 features (2 hands)** | Vocabulary is partly two-handed. Set `NUM_HANDS = 1` in `config.py` for spec-exact 63 |
| LLM = Anthropic/OpenAI | **OpenRouter** (OpenAI-compatible) + rule-based fallback | User's key; fallback keeps demos alive offline |

## Setup

```bash
pip install -r requirements.txt
```

Python 3.14, CPU-only PyTorch is fine (the model is tiny). The MediaPipe hand model
downloads automatically on first run into `models/hand_landmarker.task`.

For the LLM gloss-to-sentence step (optional — a rule-based fallback runs without it):

```bash
setx OPENROUTER_API_KEY "sk-or-..."
```

Restart the shell after `setx`. Optional overrides: `OPENROUTER_MODEL`
(default `anthropic/claude-3.5-haiku`), `OPENROUTER_BASE_URL`.

## Workflow

All commands run **from the project root** (they use `python -m ...`).

### 1. (optional) Verify the camera + landmarks
```bash
python -m common.landmarks
```
A window shows the hand skeleton on your hand. Press `q` to quit.

### 2. Record training data
```bash
python -m collect.collect_data
```
`SPACE` records a clip (after a countdown) for the current word · `n`/`b` next/prev
word · `d` delete last clip · `q` quit. Aim for ~30 clips/word, varying angle and
speed. Clips are saved to `data/raw/<WORD>/`.

Record a subset:
```bash
python -m collect.collect_data --words ME WANT WATER HELP --samples 30
```

### 3. Build the dataset
```bash
python -m preprocessing.build_dataset
```
Normalizes (wrist-relative + scale), pads to 30 frames, augments (mirror / time-warp
/ jitter), and writes `data/processed/{X,y}.npy` + `label_map.json`.

### 4. Train
```bash
python -m model.train
```
Prints train/val/test accuracy and a confusion matrix; saves the model to
`model/saved_model/`.

### 5a. Run the standalone demo (OpenCV window)
```bash
python -m inference.realtime_demo
```
Sign words; the gloss builds on screen. `SPACE` finalizes the sentence (→ LLM → spoken
aloud), `c` clears, `q` quits. A sentence also auto-finalizes after ~2.5 s of no new sign.

### 5b. …or run the web dashboard
```bash
python -m backend.app
```
Open <http://127.0.0.1:8000> — live feed with skeleton overlay, current guess +
confidence, running gloss, a "Finalize sentence" button, and a timestamped sentence log.

## Tuning (in `config.py`)

- `CONFIDENCE_THRESHOLD` (0.85) — raise if you get false detections
- `CONSECUTIVE_WINDOWS` (5) — how long a sign must persist before it's confirmed
- `SENTENCE_TIMEOUT_S` (2.5) — pause length that ends a sentence
- `VOCAB` — start with 6 words; expand and re-record/re-train to grow

If two signs are frequently confused (see the confusion matrix), record more varied
clips for that pair and retrain.

## Project layout

```
config.py               all tunable knobs (vocab, shapes, thresholds, OpenRouter)
common/landmarks.py     MediaPipe wrapper -> feature vector + annotated frame
common/preprocess.py    wrist-relative normalize + scale, pad/truncate
collect/collect_data.py webcam recorder
preprocessing/          augment.py, build_dataset.py
model/                  model.py (SignLSTM), train.py, saved_model/
inference/              engine.py, gloss_to_sentence.py, tts.py, realtime_demo.py
backend/app.py          FastAPI: MJPEG stream + WebSocket + finalize
dashboard/index.html    live dashboard
```

## Next: hardware phase

Once the pipeline feels solid, add the ESP32-S3-CAM (MJPEG stream replaces the local
webcam), the SSD1306 OLED (mirrors the gloss/sentence), the MAX98357A speaker amp
(plays the TTS audio), and the push button (wired to the same "finalize" action the
dashboard button triggers today).

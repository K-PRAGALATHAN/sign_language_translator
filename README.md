# Real-Time Sign Language → Speech Translator

A complete, working assistive-communication system that recognizes American Sign Language
(ASL) hand signs from live video, turns them into a natural spoken sentence, and delivers
that sentence as **speech + on-screen text on dedicated hardware**.

The system spans three cooperating devices that talk over WiFi:

```
┌────────────────┐      MJPEG video       ┌─────────────────────────────┐   sentence (text+audio)   ┌──────────────────────┐
│  ESP32-CAM     │ ─────over WiFi──────▶  │  Host PC / Laptop (brain)   │ ───────over WiFi───────▶  │  ESP8266 Output Node │
│  (OV3660)      │                        │  MediaPipe → LSTM → gloss   │                           │  OLED  +  Speaker    │
│  the "eyes"    │                        │  → LLM → sentence           │                           │  the "voice + face"  │
└────────────────┘                        └─────────────────────────────┘                           └──────────────────────┘
```

- **ESP32-CAM (OV3660)** — captures the signer and streams live video over WiFi.
- **Host PC** — runs the ML/AI pipeline: MediaPipe hand-landmark extraction, an LSTM sign
  classifier, gloss assembly, and an LLM that rewrites the gloss into fluent English.
- **ESP8266 (NodeMCU) output node** — receives the finished sentence and **shows it on an
  OLED** and **speaks it aloud** through a MAX98357A amplifier + speaker.

> The host PC does the heavy vision/AI work because MediaPipe and the LLM cannot run on a
> microcontroller. The ESP boards handle capture and output — exactly what they're good at.

---

## How it works (pipeline)

```
Camera (ESP32-CAM or laptop webcam)
   → MediaPipe Hands: 21 landmarks/hand  (up to 2 hands = 126 features/frame)
   → wrist-relative normalize + scale, 30-frame sliding window
   → PyTorch LSTM classifier → predicted sign + confidence
   → debounce + confidence gate → confirmed sign appended to a gloss list  e.g. [ME, WANT, WATER]
   → hold an open palm 3s (or press finalize) → sentence complete
   → OpenRouter LLM: gloss → "I want some water."
   → spoken on the ESP speaker + shown on the ESP OLED + logged on the web dashboard
```

---

## Hardware

| Component | Role |
|---|---|
| **ESP32-CAM (AI-Thinker, OV3660)** | Wireless camera — streams MJPEG to the host |
| **USB programmer / TTL adapter** | Flashing the ESP32-CAM (it has no USB port) |
| **ESP8266 NodeMCU (ESP-12F)** | Output node — drives the OLED + speaker over WiFi |
| **SSD1306 OLED (0.96", I²C)** | Displays the translated sentence |
| **MAX98357A I²S amp + speaker (4 Ω)** | Speaks the sentence aloud |
| Breadboard + jumper wires, 5 V/2 A supply | Prototyping + stable power |

### Output-node wiring (ESP8266 NodeMCU)

| OLED | NodeMCU | | MAX98357A | NodeMCU |
|---|---|---|---|---|
| VCC | 3V3 | | Vin | VIN (5V) |
| GND | G   | | GND | G |
| SDA | D2 (GPIO4) | | DIN | RX (GPIO3) |
| SCL | D1 (GPIO5) | | BCLK | D8 (GPIO15) |
|     |     | | LRC | D4 (GPIO2) |
|     |     | | SD  | 3V3 *(enable)* |

> **Power note:** the OV3660 is power-hungry — run the ESP32-CAM from a solid **5 V/2 A**
> source (a phone charger), not a weak laptop port, or the stream will freeze.

---

## Firmware (in `firmware/`)

Flashed with the **Arduino ESP32 / ESP8266 cores** via `arduino-cli` (or the Arduino IDE).
Set your WiFi SSID/password at the top of each sketch before flashing (they ship with
`YOUR_WIFI_SSID` / `YOUR_WIFI_PASSWORD` placeholders).

| Sketch | Board | Purpose |
|---|---|---|
| `firmware/CamWeb_204/` | ESP32-CAM | **Camera stream** — the working OV3660 firmware |
| `firmware/OutputNode/` | ESP8266 NodeMCU | **Output node** — OLED + speaker over TCP |
| `firmware/OledTest`, `SpeakerTest`, `CamI2CScan` | — | Bring-up / diagnostics |

### ⚠️ Critical: OV3660 needs ESP32 core **2.0.4**

The OV3660 sensor **fails to initialize on newer ESP32 cores** (2.0.17 → `0x105 not found`,
3.x → `0x106 not supported`, due to the new I²C driver). It works cleanly on the older
library. Flash the camera like this:

```bash
arduino-cli core install esp32:esp32@2.0.4
arduino-cli compile --fqbn esp32:esp32:esp32cam:PartitionScheme=huge_app firmware/CamWeb_204
arduino-cli upload  -p <PORT> --fqbn esp32:esp32:esp32cam:PartitionScheme=huge_app firmware/CamWeb_204
```

On boot the camera prints `Camera Ready! Use 'http://<ip>'`. View the raw stream at
`http://<ip>:81/stream`.

The output node (ESP8266) flashes with the standard core:

```bash
arduino-cli compile --fqbn esp8266:esp8266:nodemcuv2 firmware/OutputNode
arduino-cli upload  -p <PORT> --fqbn esp8266:esp8266:nodemcuv2 firmware/OutputNode
```

It prints its IP on boot (`OUTPUT_NODE_IP: <ip>`) — put that in `config.py`.

---

## Host software setup

Python 3.11+ (works on 3.14; TensorFlow has no 3.14 wheels, so this project uses **PyTorch**).

```bash
pip install -r requirements.txt
```

Configure the two network things in `config.py` (or via env vars):

```python
CAMERA_SOURCE   = "http://172.20.10.2:81/stream"   # ESP32-CAM stream, OR "0" for laptop webcam
OUTPUT_NODE_IP  = "172.20.10.3"                     # the ESP8266 output node's IP
```

For the LLM gloss→sentence step, put an OpenRouter key in a **`.env`** file (git-ignored):

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-haiku-4.5
```
(Without a key, a built-in rule-based rewriter is used instead.)

---

## Build & run

**1. Record training data** (per word, on the webcam or ESP32-CAM):
```bash
python -m collect.collect_data --words HELLO ME WANT WATER HELP THANK_YOU
```

**2. Build the dataset + train:**
```bash
python -m preprocessing.build_dataset
python -m model.train        # prints test accuracy + confusion matrix
```

**3. Run the full system** (web dashboard, routes output to the ESP node):
```bash
python -m backend.app
```
Open **http://localhost:8000**, sign a word, and **hold an open palm for 3 seconds** to
finish the sentence → it appears on the dashboard, shows on the ESP **OLED**, and is
**spoken by the ESP speaker**.

Prefer a no-server demo? `python -m inference.realtime_demo` runs the whole pipeline in an
OpenCV window.

Send a sentence straight to the output node to test it:
```bash
python -m inference.output_node "I want some water."
```

---

## Project layout

```
config.py                 all knobs (vocab, shapes, thresholds, camera source, node IP, OpenRouter)
common/landmarks.py       MediaPipe wrapper (webcam or ESP32-CAM stream) → features + skeleton
common/preprocess.py      wrist-relative normalize + scale, pad/truncate
collect/collect_data.py   webcam/stream data recorder
preprocessing/            augment.py, build_dataset.py
model/                    model.py (PyTorch SignLSTM), train.py
inference/                engine.py, gloss_to_sentence.py (OpenRouter+fallback),
                          tts.py, output_node.py (ESP OLED+speaker), realtime_demo.py
backend/app.py            FastAPI: MJPEG dashboard + WebSocket + routes sentences to the node
dashboard/index.html      live web dashboard
firmware/                 ESP32-CAM + ESP8266 output-node Arduino sketches
```

---

## Key technical decisions

- **Landmark-based (MediaPipe), not raw pixels** — 126 numbers/frame instead of a full
  image: robust to background/lighting and the standard approach in sign-language research.
- **LSTM over static classifiers** — most signs are *dynamic*; meaning is in the motion.
- **LLM downstream only** — gloss order (`ME WANT WATER`) isn't grammatical English; an LLM
  is ideal for the short rewrite, but is *not* used for recognition.
- **Palm-to-finish** — a 3-second open-palm hold ends a sentence, which is more reliable
  than a pure inactivity timeout.
- **ESP32 core 2.0.4 for the OV3660** — newer cores broke OV3660 detection (see above).

## License / dataset

Vocabulary/target signs follow the WLASL word list; training data here is self-recorded.
For research/educational use.

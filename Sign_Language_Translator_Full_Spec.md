# Real-Time Sign Language to Speech Translator
### Full Technical Project Specification (for implementation / Claude Code reference)

---

## 1. Project Summary

A system that recognizes ASL hand signs from live video, assembles recognized signs into a gloss sequence, converts the gloss into a natural spoken sentence using an LLM, and outputs that sentence as audio. Built around a single ESP32 microcontroller (or laptop webcam for development) with 3-4 hardware components, and an ML pipeline trained on the public WLASL dataset.

**Human input:** Live video of hand signs (webcam or ESP32-CAM)
**Output:** Spoken sentence (TTS via speaker) + text shown on OLED + dashboard log

---

## 2. High-Level Architecture

```
[Camera: ESP32-CAM or webcam]
        |
        v
[MediaPipe Hands] --> 21 landmarks (x,y,z) per frame = 63 features/frame
        |
        v
[Sliding Window Buffer] --> rolling sequence of ~30 frames (30 x 63 matrix)
        |
        v
[Trained LSTM Classifier] --> predicted sign + confidence score
        |
        v
[Debounce + Confidence Threshold Logic] --> confirms stable, non-duplicate sign
        |
        v
[Gloss Sequence Builder] --> e.g. ["ME", "WANT", "WATER"]
        |
        v
[Timeout Detector: ~2-3 sec no new sign] --> sentence considered complete
        |
        v
[LLM Prompt: gloss -> fluent English sentence]
        |
        v
[Text-to-Speech Engine] --> audio
        |
        v
[Speaker Output] + [OLED Display] + [Dashboard Log]
```

---

## 3. Hardware Components

| # | Component | Role | Notes |
|---|---|---|---|
| 1 | ESP32-S3-CAM (integrated board, OV2640 sensor, 8MB PSRAM) | Captures live hand-sign video | For dev/testing, a laptop webcam can substitute to cut hardware-latency variables while building the ML pipeline |
| 2 | Small speaker + audio amp (MAX98357A or DFPlayer Mini) | Plays translated sentence aloud | |
| 3 | OLED display (SSD1306, I2C, 128x64) | Shows gloss sequence + final sentence locally | |
| 4 (optional) | Push button | Manually confirms "end of sentence," fallback if timeout logic is unreliable | Recommended for v1 — more predictable than tuning timeout alone |

**Camera notes:** Avoid OV5640 despite higher resolution — larger frames increase latency and autofocus convergence adds delay. OV2640 at QVGA (320x240) or VGA (640x480) gives faster, more consistent frame rates, which matters more than resolution for gesture recognition. Stream via MJPEG over WiFi (same local network as backend, 5GHz preferred) to minimize latency.

---

## 4. Dataset

**Source: WLASL (Word-level American Sign Language)**
- GitHub: https://github.com/dxli94/WLASL
- Kaggle mirrors also available (search "WLASL Kaggle") — easier, avoids YouTube dead-link issues
- License: Computational Use of Data Agreement (C-UDA) — academic/non-commercial use only
- Use the **WLASL100** subset (100 most common ASL words, ~21,000 total videos across full dataset, subsets have 18-40 samples per word)

**Target vocabulary (pick words confirmed present in WLASL100):**
`HELLO, ME, YOU, WANT, NEED, WATER, FOOD, HELP, YES, NO, PLEASE, THANK YOU, SORRY, STOP, GO, HOME, MORE`
(Trim to 6-10 words for a fast MVP; expand later.)

**Dataset prep workflow:**
1. Download WLASL video files + accompanying JSON metadata (`WLASL_v0.3.json` or similar) containing gloss labels and video IDs
2. Filter JSON/video set to only the target vocabulary words
3. Run MediaPipe Hands on every frame of each filtered video clip → extract 21 landmarks (x,y,z) per frame
4. Normalize landmarks relative to wrist position (removes screen-position dependency)
5. Scale landmarks to account for hand size / distance from camera
6. Pad or truncate every sequence to a fixed length (e.g., 30 frames) for uniform LSTM input shape
7. Save processed sequences + labels as a structured dataset (e.g., NumPy arrays or a Pandas-friendly format) for training

**Optional augmentation (helps with small per-class sample counts):**
- Horizontal mirroring
- Slight time-warping (speed variation)
- Small coordinate jitter/noise

---

## 5. Model

**Input shape:** `(30 timesteps, 63 features)` per sample

**Architecture:**
```
Input (30, 63)
  -> LSTM(64, return_sequences=True)
  -> Dropout(0.3)
  -> LSTM(32)
  -> Dropout(0.3)
  -> Dense(32, activation='relu')
  -> Dense(num_classes, activation='softmax')
```

**Training setup:**
- Framework: TensorFlow/Keras (or PyTorch equivalent)
- Loss: categorical cross-entropy
- Optimizer: Adam
- Split: 70% train / 15% validation / 15% test
- Track accuracy + confusion matrix (identify commonly confused sign pairs)
- Confidence threshold for inference: ~85% (below this, treat as "no sign detected")

---

## 6. Real-Time Inference Pipeline

1. Maintain a rolling buffer of the last ~30 frames of landmark data from the live camera feed
2. Run the LSTM on this buffer at a regular interval (sliding window, not waiting for full stop)
3. If the same prediction stays above the confidence threshold across several consecutive windows, confirm it as a detected sign
4. Debounce: ignore repeated detections of the same sign while the hand remains in that position; only register a new gloss entry when the sign changes
5. Append confirmed sign to the running gloss list, e.g. `["ME", "WANT", "WATER"]`
6. **Timeout logic:** if no new sign is detected for ~2-3 seconds, treat the gloss sequence as complete and trigger the next stage
   - Fallback: physical button press can force "end of sentence" manually
7. Clear buffer and gloss list, begin next sentence

---

## 7. Gloss-to-Sentence Conversion (LLM step)

Purpose: convert raw ASL gloss order into natural, grammatically correct English before speaking it aloud. This step uses a general LLM (not for recognition — recognition is handled entirely by the MediaPipe+LSTM pipeline above; LLMs are unreliable at directly recognizing signs from raw video).

**Example prompt:**
```
Convert this ASL gloss sequence into a natural, grammatically correct
English sentence. Gloss: ME WANT WATER
```
**Expected output:**
```
I would like some water.
```

Implementation: a simple API call to any LLM (e.g., Claude API via `anthropic` SDK, or any equivalent) with the gloss string, using a short system/user prompt as above. Keep this call lightweight — it's a well-defined rewriting task, not open-ended generation.

---

## 8. Text-to-Speech Output

- Offline option: `pyttsx3` (Python) — no internet dependency, adequate quality
- Online option: Google TTS or similar — higher quality, natural voice, requires connectivity
- Output routed to the speaker/audio amp module connected via the ESP32 (or directly from the backend machine's audio output during development)

---

## 9. Output / Display Layer

- **OLED (SSD1306, I2C):** shows the current gloss sequence being built + the final translated sentence
- **Speaker:** plays the TTS audio of the final sentence
- **Dashboard (web UI):** live camera feed with hand-skeleton overlay (visual proof of MediaPipe tracking), real-time gloss being built, sentence history log with timestamps, confidence scores per detected sign

---

## 10. Software Stack

| Layer | Tool |
|---|---|
| Hand landmark detection | MediaPipe Hands |
| Data handling | NumPy, Pandas |
| Model training | TensorFlow/Keras (or PyTorch) |
| Backend / API server | Flask or FastAPI |
| Gloss-to-sentence | LLM API call (Anthropic/OpenAI/etc., simple prompt) |
| Text-to-Speech | pyttsx3 (offline) or Google TTS (online) |
| Firmware (ESP32) | Arduino IDE / PlatformIO (C++) |
| Dashboard frontend | React or plain HTML/JS with live overlay rendering |

---

## 11. Suggested Folder Structure

```
sign-language-translator/
├── data/
│   ├── raw/                  # downloaded WLASL clips (filtered subset)
│   ├── processed/            # extracted landmark sequences (npy/csv)
│   └── wlasl_metadata.json   # filtered gloss/video mapping
├── preprocessing/
│   ├── extract_landmarks.py  # MediaPipe extraction script
│   ├── normalize.py          # wrist-relative normalization + scaling
│   └── augment.py            # mirroring/jitter/time-warp
├── model/
│   ├── train.py               # LSTM training script
│   ├── model.py                # architecture definition
│   └── saved_model/           # trained weights
├── inference/
│   ├── realtime_pipeline.py   # sliding window, debounce, timeout logic
│   ├── gloss_builder.py
│   └── llm_gloss_to_sentence.py
├── tts/
│   └── speak.py
├── backend/
│   ├── app.py                  # Flask/FastAPI server
│   └── routes/
├── firmware/
│   ├── esp32_cam_stream/       # camera MJPEG streaming code
│   └── esp32_display_audio/    # OLED + speaker + button control
├── dashboard/
│   └── (React or HTML/JS frontend)
└── README.md
```

---

## 12. Development Phases

| Phase | Task | Est. Duration |
|---|---|---|
| 1 | Camera (webcam/ESP32-CAM) + MediaPipe pipeline working, landmarks visible | 1-2 days |
| 2 | Download WLASL, filter to target vocabulary, extract + preprocess landmarks | 1 day |
| 3 | Build and train LSTM model, iterate on accuracy | 1-2 days |
| 4 | Build real-time inference pipeline (sliding window, debounce, timeout) | 1-2 days |
| 5 | Integrate LLM gloss-to-sentence + TTS output | 1 day |
| 6 | Hardware integration: OLED, speaker, button | 1 day |
| 7 | Dashboard (live overlay + gloss/sentence log) | 1-2 days |
| 8 | End-to-end testing, threshold tuning, demo rehearsal | 1-2 days |

**Fast-track MVP (1 week):** limit vocabulary to 6 words, skip dashboard (or make it a single static page), use webcam instead of ESP32-CAM for the demo, use timeout + button fallback together for reliability.

---

## 13. Evaluation Metrics

- Per-sign classification accuracy (confusion matrix across vocabulary)
- Real-time inference latency (sign completion → display/speech)
- Sentence-level correctness after LLM gloss conversion (manual review)
- False positive rate (signs registered incorrectly or when none was made)

---

## 14. Key Design Decisions & Rationale (for reference)

- **Why MediaPipe landmarks instead of raw video/pixels into a model:** drastically reduces input size (63 numbers vs. full image), removes background/lighting dependency, is the standard approach used in real sign-language research (WLASL, INCLUDE benchmarks are built around pose/landmark-based methods).
- **Why not use a general-purpose multimodal LLM (GPT-4V/Gemini/Claude) for sign recognition directly:** these models are not trained at scale on sign language video, process video via sparse frame sampling (losing fine temporal motion detail), and research comparisons show weak/inconsistent performance on sign-related tasks. A dedicated landmark+LSTM pipeline is the correct, evidence-backed approach.
- **Why LSTM over a static-image classifier:** most useful signs (words, not just letters) are dynamic — meaning is encoded in motion over time, not a single pose. LSTM (or 1D-CNN/Transformer variants) captures this temporal pattern.
- **Why LLM is still used, just downstream:** gloss order (raw sign sequence) doesn't map 1:1 to natural spoken grammar. An LLM is well-suited to rephrasing a short, well-defined gloss string into fluent text — a simple text task, unlike the recognition problem.
- **Why timeout + button fallback together:** pure timeout-based sentence segmentation can be unreliable while thresholds are still being tuned; a button gives a guaranteed, demo-safe way to trigger sentence completion.
- **Why OV2640 over OV5640 for the camera sensor:** lower resolution but faster frame throughput and more mature ESP32 driver support; gesture recognition needs frame rate/motion smoothness more than image sharpness.

---

## 15. Open Items / Future Extensions

- Expand vocabulary beyond initial 6-20 words
- Incorporate facial expression / non-manual markers (important for full sign grammar, not just hand shape)
- Handle continuous fluent signing without explicit pauses (more advanced segmentation)
- On-device inference (TensorFlow Lite Micro on ESP32-S3) to reduce backend dependency
- Two-way translation: speech-to-sign avatar for the hearing person to sign back

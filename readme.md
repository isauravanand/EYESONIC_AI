# 👁️ EYSONIC AI — Smart Vision Assistive System

> **An AI-powered assistive device that detects objects, recognizes faces, identifies fire, and speaks alerts in real time — designed to help visually impaired individuals navigate safely.**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [How Each Feature Works](#how-each-feature-works)
  - [Object Detection](#1-object-detection)
  - [Face Recognition](#2-face-recognition)
  - [Fire Detection](#3-fire--flame-detection)
  - [Distance Estimation](#4-distance-estimation)
  - [Multi-Language Support](#5-multi-language-support)
  - [Voice Alerts](#6-voice-alerts)
- [API Reference](#api-reference)
- [Detection Modes](#detection-modes)
- [AI Models](#ai-models)
- [Adding a New Language](#adding-a-new-language)
- [Training a New Face](#training-a-new-face)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)

---

## Overview

EYSONIC AI is a browser-based computer vision system that streams your device's camera through an AI pipeline running on a local Flask server. Every frame is analysed for:

- **80+ object classes** using YOLOv8
- **Known faces** using dlib face encodings
- **Real fire and flame** using a 6-gate precision HSV + flicker engine
- **Distance to every detected object** using focal-length estimation

All results are spoken aloud using the browser's Web Speech API and displayed as colour-coded overlay boxes directly on the live camera feed — with no cloud dependency.

---

## Key Features

| Feature | Details |
|---------|---------|
| 🔍 Real-time object detection | YOLOv8 Nano / Small / Large — swappable from the UI |
| 👤 Face recognition | Registers and identifies known people by name |
| 🔥 Precision fire detection | 6-gate HSV + flicker algorithm; rejects skin/clothing false positives |
| 📏 Distance estimation | Focal-length formula; warns when objects enter 1m danger zone |
| 🔊 Voice alerts | Browser Web Speech API — priority: danger face → danger object → fire → nearest object |
| 🌐 Multi-language | English and Hindi (हिन्दी) — labels, directions, distances, and voice |
| 🎭 Context modes | Home, Outdoor, Crowded, Travel — filters which objects are highlighted |
| ➕ In-browser face training | Capture photo from webcam with 3-second countdown → train → instant recognition |
| 📺 HUD overlay | Marching-ant borders, confidence arcs, direction strips, animated fire boxes |
| 🚫 No internet required | Fully local — Flask + browser, no external API calls |

---

## System Architecture

```
Browser (index.html)
│
│  ┌─ Camera feed → <canvas> (video layer)
│  ├─ Overlay <canvas> (detection boxes, drawn every frame at 60fps)
│  └─ Every 3rd frame → POST /detect (640×480 JPEG, base64)
│
Flask Server (app.py)
│
│  ┌─ Face recognition  (face_recognition + dlib HOG)
│  ├─ Object detection  (YOLOv8 via Ultralytics)
│  ├─ Fire detection    (OpenCV HSV pipeline)
│  └─ JSON response → browser
│
Browser receives JSON
│  ├─ Draws boxes on overlay canvas (mapBox: server 640×480 → display px)
│  ├─ Updates sidebar panels (Objects / Faces tabs)
│  ├─ Triggers Web Speech API voice alert
│  └─ Shows fire alarm badge + strobe overlay if fire detected
```

The video canvas and overlay canvas are separate layers. Video is drawn **stretched to fill the container** on every animation frame. Detection boxes are mapped from the server's 640×480 coordinate space to the display canvas using a proportional scale function, so boxes always sit exactly on the object regardless of window size.

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Backend | Python 3.10+ / Flask | REST API server |
| Object AI | Ultralytics YOLOv8 | 80-class COCO object detection |
| Face AI | `face_recognition` + `dlib` | HOG face detection + 128-d encoding |
| Fire CV | OpenCV (HSV, morphology, contours) | Precision flame/fire detection |
| Image processing | OpenCV, NumPy, Pillow | Frame decoding, colour space conversion |
| Frontend | Vanilla HTML/CSS/JS | No framework — single file |
| Fonts | Syne (display) + JetBrains Mono (data) | Google Fonts |
| Voice | Web Speech API (`SpeechSynthesisUtterance`) | Text-to-speech in EN / HI |

---

## Project Structure

```
EYSONIC_AI/
│
├── app.py                  # Flask server — all AI pipelines live here
│
├── templates/
│   └── index.html          # Full frontend — camera, UI, canvas drawing
│
├── known_faces/            # Drop face images here (auto-loaded on start)
│   ├── ayush.jpg
│   ├── sachin.jpg
│   └── saurav.jpg
│
├── yolov8n.pt              # YOLOv8 Nano  (fastest)
├── yolov8s.pt              # YOLOv8 Small (balanced) ← default
├── yolov8l.pt              # YOLOv8 Large (most accurate)
│
├── fire.pt                 # Optional: fire-specific YOLO model
│                           # (download from Roboflow if needed)
│
├── sounds/                 # Optional: custom audio files per object
│   └── ...
│
├── debug_dlib.py           # Diagnose dlib / face_recognition install issues
├── fix_faces.py            # Auto-fix images in known_faces/ (CMYK, RGBA, etc.)
├── test_script.py          # Quick face detection test on a single image
├── SOUNDS_CONFIG.md        # Guide for adding custom audio per object
└── README.md               # This file
```

---

## Installation

### Prerequisites

- Python **3.10.x** (strict — dlib on Windows requires exactly 3.10)
- pip
- A webcam

### Step 1 — Clone or download the project

```bash
git clone https://github.com/yourname/eysonic-ai.git
cd eysonic-ai
```

### Step 2 — Install Python dependencies

```bash
pip install flask
pip install opencv-python
pip install numpy"<2.0"
pip install ultralytics
pip install Pillow
pip install face-recognition
pip install cmake dlib
```

> **Windows note:** If `dlib` fails to install, use a pre-built wheel:
> ```bash
> pip install dlib==19.24.1
> ```
> Run `python debug_dlib.py` to verify your dlib installation works correctly.

### Step 3 — Place YOLOv8 model files

The `.pt` files are included in the project. If missing, they download automatically on first run from Ultralytics servers.

### Step 4 — (Optional) Add known faces

Drop clear, front-facing JPEG/PNG photos into the `known_faces/` folder. Name each file with the person's name:

```
known_faces/
  ayush.jpg
  sachin.jpeg
  your_name.png
```

Run `python fix_faces.py` if you have any issues with images not being recognised.

---

## Running the App

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

Click **START SYSTEM** to activate the camera and begin detection.

---

## How Each Feature Works

### 1. Object Detection

YOLOv8 runs on every 3rd camera frame (to balance speed and accuracy). The frame is downsampled to 640×480 before sending to the model. Confidence thresholds are tuned per model:

| Model | Confidence threshold | Use case |
|-------|---------------------|---------|
| YOLOv8 Nano | 0.40 | Fast devices, less accurate |
| YOLOv8 Small | 0.35 | Balanced (default) |
| YOLOv8 Large | 0.25 | Most accurate, slower |

Class-agnostic NMS (`iou=0.45`) removes duplicate overlapping boxes. Detections are filtered by the active **mode** (Home / Outdoor / Crowded / Travel) for visual highlighting, but all detections are passed to the voice alert system.

### 2. Face Recognition

Uses `dlib`'s HOG face detector + a 128-dimensional face encoding per person.

**Loading:** On startup, every image in `known_faces/` is processed with `num_jitters=10` (10 random perturbations averaged) for high-quality base encodings.

**Training new faces (from browser):** The ➕ Train tab opens the front camera, runs a 3-second countdown, captures a still, and sends it to `/train_face`. The server encodes with `num_jitters=20` for maximum quality, saves the JPEG, and reloads all encodings.

**Live recognition:** Each frame is downscaled to 50% for face detection, then coordinates are scaled back. Matching uses a tolerance of `0.55` (tighter than the default 0.6 — fewer false positives). A **danger zone** triggers when a known person is estimated within 200cm.

### 3. Fire / Flame Detection

A 6-gate precision pipeline that catches real flames (lighters, candles, fires) while rejecting skin, orange clothing, and sunlit rooms.

**The 6 gates — a region must pass ALL of them:**

**Gate 1 — Skin subtraction:** Pixels matching any human skin tone (`H 0–22, S 30–170`) are removed from the candidate mask before analysis begins.

**Gate 2 — Orange surround:** Remaining pixels must match the tight orange/amber hue of flame (`H 10–28, S > 180, V > 190`). Hue starts at 10 to skip the red/skin range.

**Gate 3 — Bright core (≥ 3%):** At least 3% of the bounding box must be white/yellow (`V > 220, S < 120`) — the glowing hot centre of a real flame. Orange fabric and sunlight have no such core.

**Gate 4 — Compactness filter:** `perimeter² / area < 120`. Real flames are compact blobs. Jagged room edges, hair, and reflections fail this test.

**Gate 5 — Box fraction < 35%:** The candidate box cannot cover more than 35% of the frame. Entire orange-lit rooms are rejected.

**Gate 6 — Skin overlap < 60%:** If the candidate box overlaps with more than 60% skin pixels, it is rejected. Hands holding a lighter won't trigger — only the flame tip.

**Bonus: Local flicker scoring** — pixel-level frame difference is computed *inside* the candidate box across a 6-frame buffer. Static objects score near 0; real fire scores high. The flicker score contributes 20% of the final confidence.

**Blue base detection:** Gas lighter flames have a blue base (`H 95–135`). This range is included in the OR mask so small lighter flames are caught even when the orange tip is tiny.

**Optional YOLO fire model:** If `fire.pt` is present in the project folder (downloadable from Roboflow Universe), it runs as a parallel Stage 2 and its results are merged via NMS with the HSV candidates.

### 4. Distance Estimation

Uses the thin-lens focal length formula:

```
distance_cm = (real_object_height_cm × focal_length_px) / box_height_px
```

Focal length is estimated as `frame_width × 0.75` (calibrated for typical webcams). Real-world heights are stored in the `OBJECT_SIZES` dictionary for all 80 COCO classes. Objects within **100cm** trigger the danger flag (red border, warning voice alert).

Face distance uses average face width (20cm) instead of height for more reliable results.

### 5. Multi-Language Support

Two components work together:

**Server side (`app.py`):** The `TRANSLATIONS` dictionary maps every English label, direction, and distance phrase to each supported language. The `/detect` endpoint receives a `lang` parameter with every request and returns pre-translated fields (`label_translated`, `direction_translated`, `distance_translated`) alongside the original English values.

**Client side (`index.html`):** The `SpeechSynthesisUtterance` object uses the matching BCP-47 language tag (`en-US` or `hi-IN`). The browser automatically selects an appropriate installed voice.

The language is **stateless** — it travels as a parameter on every frame request. Switching language in the UI takes effect on the very next frame with no server restart.

**Supported languages:**

| Code | Language | Voice tag |
|------|----------|-----------|
| `en` | English | `en-US` |
| `hi` | हिन्दी (Hindi) | `hi-IN` |

### 6. Voice Alerts

Alerts fire in strict priority order (only the highest-priority alert speaks per cycle):

1. **Known person in danger zone** (within 200cm) → name + distance + direction
2. **Any known person detected** → name + distance
3. **Object in danger zone** (within 100cm) → label + direction + distance
4. **Fire detected** → "Fire detected! Danger! Please evacuate!" (5-second cooldown)
5. **Nearest other object** → label + direction + distance

A per-label cooldown of 4 seconds prevents the same alert from repeating every frame.

---

## API Reference

### `POST /detect`

Runs the full AI pipeline on a single frame.

**Request body:**
```json
{
  "image":       "data:image/jpeg;base64,/9j/...",
  "mode":        "home",
  "model":       "yolov8s",
  "lang":        "en",
  "sensitivity": "medium"
}
```

**Response:**
```json
{
  "visual_detections": [...],
  "audio_detections":  [...],
  "face_detections":   [...],
  "fire_detections":   [...],
  "has_face_danger":   false,
  "has_fire":          false,
  "has_danger_fire":   false,
  "lang":              "en"
}
```

Each detection object contains:
```json
{
  "label":               "chair",
  "label_translated":    "कुर्सी",
  "direction":           "on your left",
  "direction_translated":"आपके बाईं तरफ",
  "distance":            "near",
  "distance_translated": "पास",
  "distance_cm":         180,
  "distance_text":       "1.8m",
  "box":                 [x1, y1, x2, y2],
  "danger":              false,
  "confidence":          0.87
}
```

---

### `POST /train_face`

Registers a new face encoding from a captured image.

**Request:**
```json
{ "name": "Ayush", "image": "data:image/jpeg;base64,..." }
```

**Response:**
```json
{ "success": true, "message": "✓ Face trained for: Ayush", "known_faces_count": 3, "known_faces": ["Ayush", "Sachin", "Saurav"] }
```

---

### `GET /list_known_faces`

Returns all currently registered face names.

---

### `GET /languages`

Returns available language options.

---

### `GET /status`

Returns server health, loaded models, and supported languages.

---

## Detection Modes

| Mode | Objects highlighted |
|------|-------------------|
| 🏠 Home | Chair, bed, TV, bottle, cup, couch, remote, person, dog, cat, dining table, potted plant |
| 🌳 Outdoor | Car, bus, dog, person, bicycle, traffic light, stop sign, motorcycle, truck, bench, umbrella |
| 👥 Crowded | Person, backpack, handbag, car, bicycle, cell phone |
| ✈️ Travel | Suitcase, backpack, handbag, bus, train, car, airplane, person, bench |

> **Note:** Only highlighted objects get visual bounding boxes. All detected objects (including non-mode ones) still trigger voice alerts.

---

## AI Models

| File | Size | Speed | Accuracy | Best for |
|------|------|-------|---------|---------|
| `yolov8n.pt` | 6 MB | ⚡⚡⚡ Fastest | ★★☆☆ | Low-power devices |
| `yolov8s.pt` | 22 MB | ⚡⚡ Fast | ★★★☆ | Default — best balance |
| `yolov8l.pt` | 87 MB | ⚡ Slower | ★★★★ | High accuracy required |
| `fire.pt` | varies | ⚡⚡ Fast | ★★★★ | Optional fire-specific model |

All models detect the **80 COCO object classes** (person, car, chair, bottle, dog, etc.).

To use a fire-specific YOLO model, download one from [Roboflow Universe](https://universe.roboflow.com) (search "fire detection YOLOv8"), name it `fire.pt`, and place it in the project root. It will be loaded automatically.

---

## Adding a New Language

**Step 1 — Add translations to `app.py`:**

```python
TRANSLATIONS["ta"] = {
    "on your left":   "உங்கள் இடதுபுறம்",
    "on your right":  "உங்கள் வலதுபுறம்",
    "straight ahead": "நேரே முன்னே",
    "very close":     "மிக அருகில்",
    "near":           "அருகில்",
    "far":            "தூரத்தில்",
    "person":         "நபர்",
    "car":            "கார்",
    "fire":           "தீ",
    # ... add all labels
}
```

**Step 2 — Add to the language modal in `index.html`:**

```html
<div class="lopt" id="lopt-ta" onclick="setLang('ta')">
  <div class="lflag">🇮🇳</div>
  <div>
    <div class="lname">தமிழ்</div>
    <div class="lnat">Tamil · தமிழ் மொழி</div>
  </div>
</div>
```

**Step 3 — Add the BCP-47 voice tag in `setLang()`:**

```javascript
u.lang = currentLang === 'hi' ? 'hi-IN'
       : currentLang === 'ta' ? 'ta-IN'
       : 'en-US';
```

No server restart required — the change applies on the next frame.

---

## Training a New Face

### Method 1 — In-browser (recommended)

1. Open the app and click **START SYSTEM**
2. Go to the **➕ Train** tab in the sidebar
3. Enter the person's name
4. Click **Open Camera** — the front camera activates (mirrored like a selfie)
5. Position face in frame, click **Snap (3s)** — countdown runs, photo is taken
6. Click **Train This Face** — the server encodes and saves the face
7. Switch to the **👤 Faces** tab to confirm the name appears

### Method 2 — Drop a photo into the folder

1. Place a clear, front-facing JPEG in `known_faces/`
2. Name it `firstname_lastname.jpg` (underscores become spaces, title-cased)
3. Restart the server — faces are loaded at startup

**Tips for best recognition accuracy:**
- Use a clear, well-lit photo with one face only
- Avoid heavy shadows, side profiles, or sunglasses
- Multiple photos of the same person = run Method 2 multiple times with different angles, or contribute averaged encodings manually

---

## Troubleshooting

### Face not detected when training

Run the diagnostic tool:
```bash
python debug_dlib.py
```

If it fails, reinstall dlib:
```bash
pip uninstall dlib face-recognition -y
pip install dlib==19.24.1 face-recognition
```

### Images in `known_faces/` not loading

Run the auto-fix tool:
```bash
python fix_faces.py
```

This converts all images to clean RGB JPEGs and reports which ones have no detectable face.

### Fire false positives (detecting skin/clothing as fire)

The v2 precision engine should handle this. If you still see false positives:
- Ensure your camera white-balance is set to auto (avoid warm tungsten presets)
- The blue-range in the HSV orange surround starts at H=10 to skip red/skin — if your lighting is unusually warm, you can raise the H lower bound to `12` in `_orange_surround_mask()`

### Boxes are in wrong position / only half the screen used

This was a known bug (fixed in v2). Both canvases must be sized to `cam-wrap.clientWidth × clientHeight` and the video drawn with `drawImage(vid, 0, 0, dW, dH)` (stretched fill, no letterbox). Detection coordinates are mapped from 640×480 server space using `mapBox()`.

### `dlib` import error on NumPy 2.x

```bash
pip install "numpy<2.0"
```

dlib is not yet compatible with NumPy 2.x.

---

## Known Limitations

- **Distance estimation is approximate** — the focal-length formula assumes a fixed camera FOV. Accuracy degrades beyond 5 metres and for objects at extreme angles.
- **Face recognition degrades** with masks, heavy makeup, extreme side profiles, or very low-resolution captures.
- **Fire detection** using HSV requires reasonable lighting. In very dark environments, the brightness gates may miss low-intensity flames. Use a fire-specific YOLO model (`fire.pt`) for better night performance.
- **Voice output** requires the OS to have the language's voice installed. Hindi voice (`hi-IN`) may not be available on all systems — check `speechSynthesis.getVoices()` in your browser console.
- **Performance** on CPU-only machines: YOLOv8 Nano runs comfortably at 10–15 FPS. Large model may drop below 5 FPS. Face recognition adds ~80ms per frame on CPU.

---

## Credits

Built by **Team EYSONIC** as an AI-powered assistive vision device for the visually impaired.

- Object detection: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Face recognition: [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)
- Computer vision: [OpenCV](https://opencv.org)
- Web server: [Flask](https://flask.palletsprojects.com)

---

*EYSONIC AI v2.0 — See the world through AI* 👁️🔊
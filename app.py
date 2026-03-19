from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
import face_recognition
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  LANGUAGE TRANSLATIONS
# ─────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "en": {
        # Directions
        "on your left":    "on your left",
        "on your right":   "on your right",
        "straight ahead":  "straight ahead",
        # Distances
        "very close":      "very close",
        "near":            "near",
        "far":             "far",
        # YOLO object labels (COCO 80 classes, most common)
        "person":          "person",
        "bicycle":         "bicycle",
        "car":             "car",
        "motorcycle":      "motorcycle",
        "airplane":        "airplane",
        "bus":             "bus",
        "train":           "train",
        "truck":           "truck",
        "boat":            "boat",
        "traffic light":   "traffic light",
        "fire hydrant":    "fire hydrant",
        "stop sign":       "stop sign",
        "parking meter":   "parking meter",
        "bench":           "bench",
        "bird":            "bird",
        "cat":             "cat",
        "dog":             "dog",
        "horse":           "horse",
        "sheep":           "sheep",
        "cow":             "cow",
        "elephant":        "elephant",
        "bear":            "bear",
        "zebra":           "zebra",
        "giraffe":         "giraffe",
        "backpack":        "backpack",
        "umbrella":        "umbrella",
        "handbag":         "handbag",
        "tie":             "tie",
        "suitcase":        "suitcase",
        "frisbee":         "frisbee",
        "skis":            "skis",
        "snowboard":       "snowboard",
        "sports ball":     "sports ball",
        "kite":            "kite",
        "baseball bat":    "baseball bat",
        "baseball glove":  "baseball glove",
        "skateboard":      "skateboard",
        "surfboard":       "surfboard",
        "tennis racket":   "tennis racket",
        "bottle":          "bottle",
        "wine glass":      "wine glass",
        "cup":             "cup",
        "fork":            "fork",
        "knife":           "knife",
        "spoon":           "spoon",
        "bowl":            "bowl",
        "banana":          "banana",
        "apple":           "apple",
        "sandwich":        "sandwich",
        "orange":          "orange",
        "broccoli":        "broccoli",
        "carrot":          "carrot",
        "hot dog":         "hot dog",
        "pizza":           "pizza",
        "donut":           "donut",
        "cake":            "cake",
        "chair":           "chair",
        "couch":           "couch",
        "potted plant":    "potted plant",
        "bed":             "bed",
        "dining table":    "dining table",
        "toilet":          "toilet",
        "tv":              "tv",
        "laptop":          "laptop",
        "mouse":           "mouse",
        "remote":          "remote",
        "keyboard":        "keyboard",
        "cell phone":      "cell phone",
        "microwave":       "microwave",
        "oven":            "oven",
        "toaster":         "toaster",
        "sink":            "sink",
        "refrigerator":    "refrigerator",
        "book":            "book",
        "clock":           "clock",
        "vase":            "vase",
        "scissors":        "scissors",
        "teddy bear":      "teddy bear",
        "hair drier":      "hair drier",
        "toothbrush":      "toothbrush",
        # Special face labels
        "Unknown Person":  "Unknown Person",
        # Alert phrases
        "detected":        "detected",
        "danger":          "Warning!",
        "fire":            "fire",
        "smoke":           "smoke",
        "flame":           "flame",
    },
    "hi": {
        # Directions (Hindi)
        "on your left":    "आपके बाईं तरफ",
        "on your right":   "आपके दाईं तरफ",
        "straight ahead":  "सीधे आगे",
        # Distances (Hindi)
        "very close":      "बहुत पास",
        "near":            "पास",
        "far":             "दूर",
        # YOLO object labels (Hindi)
        "person":          "व्यक्ति",
        "bicycle":         "साइकिल",
        "car":             "कार",
        "motorcycle":      "मोटरसाइकिल",
        "airplane":        "हवाई जहाज",
        "bus":             "बस",
        "train":           "ट्रेन",
        "truck":           "ट्रक",
        "boat":            "नाव",
        "traffic light":   "ट्रैफिक लाइट",
        "fire hydrant":    "अग्निशामक नल",
        "stop sign":       "रुकने का संकेत",
        "parking meter":   "पार्किंग मीटर",
        "bench":           "बेंच",
        "bird":            "पक्षी",
        "cat":             "बिल्ली",
        "dog":             "कुत्ता",
        "horse":           "घोड़ा",
        "sheep":           "भेड़",
        "cow":             "गाय",
        "elephant":        "हाथी",
        "bear":            "भालू",
        "zebra":           "ज़ेबरा",
        "giraffe":         "जिराफ",
        "backpack":        "बैगपैक",
        "umbrella":        "छाता",
        "handbag":         "हैंडबैग",
        "tie":             "टाई",
        "suitcase":        "सूटकेस",
        "frisbee":         "फ्रिसबी",
        "skis":            "स्की",
        "snowboard":       "स्नोबोर्ड",
        "sports ball":     "खेल की गेंद",
        "kite":            "पतंग",
        "baseball bat":    "बेसबॉल बैट",
        "baseball glove":  "बेसबॉल दस्ताना",
        "skateboard":      "स्केटबोर्ड",
        "surfboard":       "सर्फबोर्ड",
        "tennis racket":   "टेनिस रैकेट",
        "bottle":          "बोतल",
        "wine glass":      "वाइन ग्लास",
        "cup":             "कप",
        "fork":            "काँटा",
        "knife":           "चाकू",
        "spoon":           "चम्मच",
        "bowl":            "कटोरा",
        "banana":          "केला",
        "apple":           "सेब",
        "sandwich":        "सैंडविच",
        "orange":          "संतरा",
        "broccoli":        "ब्रोकली",
        "carrot":          "गाजर",
        "hot dog":         "हॉट डॉग",
        "pizza":           "पिज़्ज़ा",
        "donut":           "डोनट",
        "cake":            "केक",
        "chair":           "कुर्सी",
        "couch":           "सोफा",
        "potted plant":    "गमले का पौधा",
        "bed":             "बिस्तर",
        "dining table":    "खाने की मेज",
        "toilet":          "शौचालय",
        "tv":              "टेलीविजन",
        "laptop":          "लैपटॉप",
        "mouse":           "माउस",
        "remote":          "रिमोट",
        "keyboard":        "कीबोर्ड",
        "cell phone":      "मोबाइल फोन",
        "microwave":       "माइक्रोवेव",
        "oven":            "ओवन",
        "toaster":         "टोस्टर",
        "sink":            "सिंक",
        "refrigerator":    "फ्रिज",
        "book":            "किताब",
        "clock":           "घड़ी",
        "vase":            "फूलदान",
        "scissors":        "कैंची",
        "teddy bear":      "टेडी बेयर",
        "hair drier":      "हेयर ड्रायर",
        "toothbrush":      "टूथब्रश",
        # Special face labels
        "Unknown Person":  "अज्ञात व्यक्ति",
        # Alert phrases
        "detected":        "पहचाना गया",
        "danger":          "खतरा!",
        "fire":            "आग",
        "smoke":           "धुआँ",
        "flame":           "लपट",
    }
}

def translate(key, lang="en"):
    """Translate a label/phrase to the target language."""
    lang_map = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return lang_map.get(key, key)

def translate_detection(detection, lang):
    """Return translated label, direction, distance for a detection dict."""
    raw_label     = detection.get("raw_label", detection.get("label", ""))
    raw_direction = detection.get("direction", "")
    raw_distance  = detection.get("distance", "")

    return {
        **detection,
        "label_translated":     translate(raw_label, lang),
        "direction_translated": translate(raw_direction, lang),
        "distance_translated":  translate(raw_distance, lang),
    }

# ─────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────

def safe_bgr_to_rgb(frame):
    if frame is None:
        return None
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)

app = Flask(__name__)

# ── Model registry ──────────────────────────────────────────
models = {}
model_files = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "fire":    "fire.pt",      # optional fire-specific model
}

# ── Face registry ────────────────────────────────────────────
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    known_faces_dir = "known_faces"
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        print("✓ Created known_faces directory")
        return

    known_face_encodings = []
    known_face_names = []

    try:
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(known_faces_dir, filename)
                try:
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(filepath).convert("RGB")
                    rgb_frame = np.ascontiguousarray(np.array(pil_img, dtype=np.uint8))

                    # ── Accuracy boost: use CNN model if GPU available,
                    #    otherwise HOG with multiple upsamples ─────────
                    face_locations = face_recognition.face_locations(
                        rgb_frame, model="hog", number_of_times_to_upsample=2
                    )
                    face_encodings = face_recognition.face_encodings(
                        rgb_frame, face_locations, num_jitters=10
                    )

                    if face_encodings:
                        name = os.path.splitext(filename)[0].replace('_', ' ').title()
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(name)
                        print(f"✓ Loaded face: {name}")
                    else:
                        print(f"⚠ No face found in {filename}")
                except Exception as img_e:
                    print(f"⚠ Error processing {filename}: {img_e}")

        print(f"✓ Face recognition ready: {len(known_face_names)} known faces loaded")
    except ImportError:
        print("⚠ face_recognition not installed.")
    except Exception as e:
        print(f"⚠ Error loading known faces: {e}")


def train_face_from_image(name, image_b64):
    try:
        if not name or name.strip() == "":
            return False, "Name cannot be empty"
        try:
            header, encoded = image_b64.split(",", 1)
        except Exception:
            return False, "Invalid base64 format"

        image_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if frame is None:
            return False, "Failed to decode image"
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        rgb_frame = safe_bgr_to_rgb(frame)
        rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

        # High-quality encoding for training (more jitters = better accuracy)
        face_locations = face_recognition.face_locations(
            rgb_frame, model="hog", number_of_times_to_upsample=2
        )
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations, num_jitters=20
        )

        if not face_encodings:
            return False, "No face detected in the image"
        if len(face_encodings) > 1:
            return False, f"Multiple faces detected ({len(face_encodings)}). Use a single face image."

        known_faces_dir = "known_faces"
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)

        filename = name.strip().replace(' ', '_') + ".jpg"
        filepath = os.path.join(known_faces_dir, filename)
        cv2.imwrite(filepath, frame)
        load_known_faces()
        return True, f"✓ Face trained successfully for: {name}"

    except Exception as e:
        return False, f"Error training face: {str(e)}"


load_known_faces()

# ── Model loading ────────────────────────────────────────────
def load_model(model_name):
    if model_name not in models:
        model_file = model_files.get(model_name, "yolov8n.pt")
        try:
            print(f"Loading model: {model_file}")
            models[model_name] = YOLO(model_file)
            print(f"✓ Model loaded: {model_name}")
        except Exception as e:
            print(f"⚠ Error loading model {model_name}: {e}")
            if model_name != "yolov8n":
                print("Falling back to yolov8n")
                return load_model("yolov8n")
            return None
    return models[model_name]

load_model("yolov8n")

# ── Object size reference (cm) ───────────────────────────────
OBJECT_SIZES = {
    "person": 170, "chair": 80, "bottle": 25, "cup": 10, "bed": 200,
    "couch": 80, "tv": 60, "car": 450, "bus": 1000, "bicycle": 180,
    "dog": 60, "cat": 30, "backpack": 50, "handbag": 30, "suitcase": 70,
    "train": 2600, "airplane": 2400, "truck": 800, "motorcycle": 200,
    "traffic light": 300, "stop sign": 120, "remote": 15,
    "cell phone": 15, "laptop": 35, "keyboard": 45, "mouse": 12,
    "book": 25, "clock": 30, "vase": 25, "umbrella": 100,
    "dining table": 150, "bench": 150, "horse": 160, "cow": 150,
    "elephant": 280, "bear": 180,
    # Fire / safety
    "fire": 50, "smoke": 200, "flame": 50,
}

def estimate_distance_cm(label, box, frame_width):
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    focal_length = frame_width * 0.75          # Slightly improved focal estimate
    object_height = OBJECT_SIZES.get(label, 30)
    if box_height > 0:
        distance_cm = int((object_height * focal_length) / box_height)
    else:
        distance_cm = 500
    return max(1, min(distance_cm, 5000))

def estimate_face_distance_cm(face_box, frame_width):
    left, top, right, bottom = face_box
    face_width = right - left
    focal_length = frame_width * 0.75
    if face_width > 0:
        distance_cm = int((20 * focal_length) / face_width)
    else:
        distance_cm = 500
    return max(1, min(distance_cm, 3000))

def get_distance_label(distance_cm):
    if distance_cm < 100:
        return "very close"
    elif distance_cm < 300:
        return "near"
    else:
        return "far"

# ── Detection modes ──────────────────────────────────────────
MODES = {
    "home":    ["chair", "bed", "tv", "bottle", "cup", "couch", "remote",
                "person", "dog", "cat", "dining table", "potted plant"],
    "outdoor": ["car", "bus", "dog", "person", "bicycle", "traffic light",
                "stop sign", "motorcycle", "truck", "bench", "umbrella"],
    "crowded": ["person", "backpack", "handbag", "car", "bicycle", "cell phone"],
    "travel":  ["suitcase", "backpack", "handbag", "bus", "train", "car",
                "airplane", "person", "bench"],
}



# ─────────────────────────────────────────────────────────────
#  PRECISION FIRE / FLAME DETECTION ENGINE  v2
#
#  Problem solved: old HSV ranges matched skin, clothing, sunlight.
#
#  Key insight — a real lighter / candle flame has ALL of these:
#    1. A bright white/yellow CORE (V > 230, S < 80 at centre)
#    2. An orange surround  (H 10-28, S > 180, V > 200)
#    3. A blue base (lighter flames only — H 100-130)
#    4. Compact, vertically-oriented shape
#    5. Local brightness PEAK — centre brighter than surrounding ring
#    6. Temporal flicker WITHIN the candidate box
#
#  Things that are NOT fire:
#    • Skin: medium S, medium V, H 0-22 → excluded by white-core check
#    • Orange fabric: uniform colour, no bright core, no flicker
#    • Sunlight/glare: extremely large, aspect ratio too wide
#    • Red LEDs: no orange surround, no core gradient
# ─────────────────────────────────────────────────────────────

import collections
_fire_buf = collections.deque(maxlen=6)   # per-frame gray ROIs keyed by box position

_fire_model = None
_fire_model_checked = False

def get_fire_model():
    global _fire_model, _fire_model_checked
    if _fire_model_checked:
        return _fire_model
    _fire_model_checked = True
    for p in ["fire.pt", "fire_model.pt", "models/fire.pt"]:
        if os.path.exists(p):
            try:
                _fire_model = YOLO(p)
                print(f"✓ Fire YOLO model loaded: {p}")
                return _fire_model
            except Exception as e:
                print(f"⚠ {p} failed: {e}")
    print("ℹ No fire.pt — using precision HSV+core+flicker pipeline")
    return None


def _skin_mask(hsv):
    """Return mask of skin-coloured pixels to subtract from fire candidates."""
    # Skin: H 0-22, S 30-170, V 60-255  (covers all skin tones)
    return cv2.inRange(hsv,
        np.array([0,  30,  60]),
        np.array([22, 170, 255])
    )


def _flame_core_mask(hsv, gray):
    """
    Bright white-yellow core of a flame.
    H 15-45, S 0-120 (almost white/yellow), V > 220
    Also catches pure white hotspot (S < 30, V > 240).
    """
    yellow_core = cv2.inRange(hsv,
        np.array([15,  0, 220]),
        np.array([45, 120, 255])
    )
    white_core = cv2.inRange(hsv,
        np.array([0,  0, 240]),
        np.array([180, 30, 255])
    )
    return yellow_core | white_core


def _orange_surround_mask(hsv):
    """
    The orange/amber body of a flame.
    Tight HSV range — high saturation, high brightness, orange hue only.
    Explicitly AVOIDS the skin hue range (0-12).
    """
    return cv2.inRange(hsv,
        np.array([10, 180, 190]),   # H starts at 10 to skip red/skin
        np.array([28, 255, 255])
    )


def _blue_base_mask(hsv):
    """Blue base of lighter/gas flame."""
    return cv2.inRange(hsv,
        np.array([95, 120, 120]),
        np.array([135, 255, 255])
    )


def _local_flicker(gray, box, prev_frames):
    """
    Measure pixel-level change INSIDE the candidate box across stored frames.
    Returns flicker score 0-1. Static objects score near 0.
    """
    x1, y1, x2, y2 = box
    x1, y1 = max(0,x1), max(0,y1)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    diffs = []
    for prev_gray in prev_frames:
        if prev_gray.shape == gray.shape:
            prev_roi = prev_gray[y1:y2, x1:x2]
            if prev_roi.shape == roi.shape:
                d = cv2.absdiff(roi, prev_roi)
                diffs.append(float(d.mean()))

    if not diffs:
        return 0.5   # neutral on first frames
    mean_diff = float(np.mean(diffs))
    # Fire flickers at ~8-25 mean pixel diff; static objects < 3
    return float(np.clip((mean_diff - 3.0) / 18.0, 0.0, 1.0))


def _brightness_peak_score(gray, box):
    """
    Check that centre of box is brighter than its border ring.
    Real flames are brightest at centre; skin/fabric is uniform.
    Returns score 0-1.
    """
    x1, y1, x2, y2 = box
    x1,y1 = max(0,x1), max(0,y1)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    h, w = roi.shape
    if h < 6 or w < 6:
        return 0.5

    # Centre 50%
    cy1, cy2 = h//4, 3*h//4
    cx1, cx2 = w//4, 3*w//4
    centre_mean = float(roi[cy1:cy2, cx1:cx2].mean())

    # Border ring
    border = np.concatenate([
        roi[:cy1,:].flatten(), roi[cy2:,:].flatten(),
        roi[:,: cx1].flatten(), roi[:,cx2:].flatten()
    ])
    if border.size == 0:
        return 0.5
    border_mean = float(border.mean())

    ratio = (centre_mean - border_mean) / max(border_mean, 1.0)
    return float(np.clip(ratio / 0.25, 0.0, 1.0))   # 0.25 brightness ratio = score 1.0


def detect_fire_in_frame(frame_bgr, lang="en"):
    """
    Precision fire detection:
      1. Build fire-pixel mask (core + orange surround) minus skin
      2. Find contours → apply shape/size filters
      3. For each candidate: score brightness peak + local flicker
      4. Final score = color × core_present × brightness_peak × flicker
      5. Optional: YOLO fire model if fire.pt exists
    """
    h, w = frame_bgr.shape[:2]
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Store frame for flicker check
    _fire_buf.append(gray.copy())
    prev_frames = list(_fire_buf)[:-1]  # all but current

    # ── Build combined fire mask ──────────────────────────────
    skin      = _skin_mask(hsv)
    core      = _flame_core_mask(hsv, gray)
    orange    = _orange_surround_mask(hsv)
    blue_base = _blue_base_mask(hsv)

    # Fire = (orange surround OR blue base) that is NOT skin
    # Core must be present nearby to avoid false positives on orange objects
    raw_fire = cv2.bitwise_or(orange, blue_base)
    raw_fire = cv2.bitwise_and(raw_fire, cv2.bitwise_not(skin))

    # Dilate core slightly and require overlap with raw fire
    core_dilated = cv2.dilate(core, np.ones((9,9),np.uint8), iterations=3)
    fire_mask = cv2.bitwise_and(raw_fire, core_dilated)

    # Morphological cleanup — remove isolated noise pixels
    k3 = np.ones((3,3), np.uint8)
    k5 = np.ones((5,5), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN,  k3)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, k5)
    fire_mask = cv2.dilate(fire_mask, k3, iterations=1)

    candidates = []

    # ── YOLO fire model (if available) — highest priority ─────
    fire_model = get_fire_model()
    if fire_model is not None:
        try:
            yolo_res = fire_model(frame_bgr, conf=0.30, iou=0.4, verbose=False)
            for r in yolo_res:
                for box in r.boxes:
                    name = fire_model.names.get(int(box.cls[0]), "fire").lower()
                    if name not in ("fire","smoke","flame"): name = "fire"
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    flicker = _local_flicker(gray, [x1,y1,x2,y2], prev_frames)
                    conf = float(box.conf[0]) * (0.5 + 0.5*flicker)
                    candidates.append({
                        "box": [x1,y1,x2,y2], "confidence": round(conf,2),
                        "label": name, "sub_label": "YOLO", "source": "yolo"
                    })
        except Exception as e:
            print(f"⚠ fire model: {e}")

    # ── HSV precision candidates ──────────────────────────────
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:        # min 80px² — catches small lighter flame
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        # ── Shape filters ─────────────────────────────────────
        # 1. Not too wide (sunlight patches are very wide)
        aspect = cw / max(ch, 1)
        if aspect > 3.5:
            continue

        # 2. Not too large relative to frame (covers >35% = probably not a real flame)
        box_frac = (cw * ch) / max(w * h, 1)
        if box_frac > 0.35:
            continue

        # 3. Compactness — perimeter²/area (circle=12.6, elongated>40)
        perim = cv2.arcLength(cnt, True)
        compact = (perim**2) / max(area, 1)
        if compact > 120:    # too jagged / elongated
            continue

        # ── Score components ──────────────────────────────────
        # A. Coverage — fire pixels / contour area
        roi_mask = fire_mask[y:y+ch, x:x+cw]
        coverage = float(np.count_nonzero(roi_mask)) / max(cw*ch, 1)
        if coverage < 0.20:   # at least 20% of bounding box must be fire-colored
            continue

        # B. Core present — white/yellow hot-spot inside the box
        core_roi = core[y:y+ch, x:x+cw]
        core_frac = float(np.count_nonzero(core_roi)) / max(cw*ch, 1)
        if core_frac < 0.03:   # must have ≥3% bright core — BLOCKS skin/fabric
            continue

        # C. Brightness peak at centre
        bp = _brightness_peak_score(gray, [x, y, x+cw, y+ch])

        # D. Local flicker (temporal)
        flicker = _local_flicker(gray, [x, y, x+cw, y+ch], prev_frames)

        # ── Combine into final confidence ─────────────────────
        # Coverage and core_frac are gates (must pass), then scored
        color_score  = min(coverage * 2.0, 1.0)       # 0-1
        core_score   = min(core_frac * 8.0, 1.0)      # 0-1
        # Flicker and brightness only matter if we have enough temporal data
        motion_score = flicker if len(prev_frames) >= 2 else 0.5
        bright_score = bp

        conf = (color_score  * 0.30 +
                core_score   * 0.35 +
                motion_score * 0.20 +
                bright_score * 0.15)

        if conf < 0.30:
            continue

        # Reject if this looks like it overlaps heavily with a skin region
        skin_roi = skin[y:y+ch, x:x+cw]
        skin_frac = float(np.count_nonzero(skin_roi)) / max(cw*ch, 1)
        if skin_frac > 0.60:   # >60% skin = false positive (face/arm)
            continue

        candidates.append({
            "box":        [x, y, x+cw, y+ch],
            "confidence": round(conf, 2),
            "label":      "fire",
            "sub_label":  "HSV",
            "source":     "hsv",
            "area":       int(area),
        })

    # ── NMS — merge overlapping candidates ────────────────────
    if len(candidates) > 1:
        bxywh  = [[c["box"][0], c["box"][1],
                   c["box"][2]-c["box"][0],
                   c["box"][3]-c["box"][1]] for c in candidates]
        scores = [c["confidence"] for c in candidates]
        try:
            idx = cv2.dnn.NMSBoxes(bxywh, scores, 0.28, 0.45)
            if len(idx) > 0:
                keep = [int(i) for i in idx.flatten()]
                candidates = [candidates[i] for i in keep]
        except Exception:
            pass

    # ── Convert to API response format ────────────────────────
    final = []
    for c in candidates:
        x1, y1, x2, y2 = c["box"]
        cx = (x1+x2)//2
        direction = ("on your left"  if cx < w*0.33
                     else "on your right" if cx > w*0.66
                     else "straight ahead")
        box_frac  = ((x2-x1)*(y2-y1)) / max(w*h, 1)
        is_danger = box_frac > 0.015 or c["confidence"] > 0.75
        final.append({
            "label":               c["label"],
            "raw_label":           c["label"],
            "label_translated":    translate(c["label"], lang),
            "direction":           direction,
            "direction_translated": translate(direction, lang),
            "distance":            "very close" if is_danger else "near",
            "distance_translated": translate("very close" if is_danger else "near", lang),
            "distance_cm":         40 if is_danger else 120,
            "distance_text":       "nearby" if is_danger else "detected",
            "box":                 [x1, y1, x2, y2],
            "danger":              is_danger,
            "confidence":          c["confidence"],
            "is_fire":             True,
            "source":              c.get("source","hsv"),
            "sub_label":           c.get("sub_label",""),
        })

    return final

# ── Accuracy settings per model ──────────────────────────────
#  Higher-capacity models use slightly lower conf thresholds to
#  catch more true positives they can reliably identify.
MODEL_CONF = {
    "yolov8n": 0.40,   # nano  – needs higher bar to avoid FP
    "yolov8s": 0.35,
    "yolov8m": 0.30,
    "yolov8l": 0.25,   # large – most accurate, lower threshold OK
}

# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "running",
        "models_loaded": list(models.keys()),
        "available_models": list(model_files.keys()),
        "supported_languages": list(TRANSLATIONS.keys()),
    })

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data       = request.json
        mode       = data.get('mode', 'home')
        model_name = data.get('model', 'yolov8n')
        image_b64  = data.get('image')
        lang       = data.get('lang', 'en')          # ← NEW: language selection

        if lang not in TRANSLATIONS:
            lang = 'en'

        if not image_b64:
            return jsonify({"error": "No image provided"}), 400

        model = load_model(model_name)
        if model is None:
            return jsonify({"error": f"Failed to load model: {model_name}"}), 500

        header, encoded = image_b64.split(",", 1)
        image_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        h, w, _ = frame.shape
        visual_detections = []
        audio_detections  = []
        face_detections   = []

        # ── FACE RECOGNITION ──────────────────────────────────
        try:
            rgb_frame  = safe_bgr_to_rgb(frame)
            # Downscale to 50% (not 25%) for better small-face recall
            scale = 0.5
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=scale, fy=scale)
            small_frame = np.ascontiguousarray(small_frame, dtype=np.uint8)

            face_locations = face_recognition.face_locations(
                small_frame, model="hog", number_of_times_to_upsample=1
            )
            face_encodings = face_recognition.face_encodings(
                small_frame, face_locations, num_jitters=1
            )

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                top    = int(top    / scale)
                right  = int(right  / scale)
                bottom = int(bottom / scale)
                left   = int(left   / scale)

                matches        = face_recognition.compare_faces(
                    known_face_encodings, face_encoding, tolerance=0.55   # tighter = fewer FP
                )
                name           = "Unknown Person"
                confidence     = 0.0

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_idx = np.argmin(face_distances)
                    if matches[best_idx]:
                        name       = known_face_names[best_idx]
                        confidence = 1.0 - float(face_distances[best_idx])

                face_distance_cm   = estimate_face_distance_cm([left, top, right, bottom], w)
                face_distance_text = (
                    f"{face_distance_cm / 100:.1f}m"
                    if face_distance_cm >= 100
                    else f"{face_distance_cm}cm"
                )
                is_danger = (name != "Unknown Person" and face_distance_cm < 200)

                # Translate name for Unknown Person
                display_name = translate("Unknown Person", lang) if name == "Unknown Person" else name

                face_detections.append({
                    "name":          name,
                    "name_display":  display_name,
                    "box":           [left, top, right, bottom],
                    "confidence":    round(confidence, 2),
                    "is_known":      name != "Unknown Person",
                    "distance_cm":   face_distance_cm,
                    "distance_text": face_distance_text,
                    "is_danger":     is_danger,
                })
        except Exception as e:
            print(f"⚠ Face recognition error: {e}")

        # ── OBJECT DETECTION (YOLO) ───────────────────────────
        conf_threshold = MODEL_CONF.get(model_name, 0.35)

        results = model(
            frame,
            conf=conf_threshold,
            iou=0.45,          # NMS IoU – reduce duplicate boxes
            verbose=False,
            agnostic_nms=True, # class-agnostic NMS for crowded scenes
        )

        for r in results:
            for box in r.boxes:
                cls   = int(box.cls[0])
                label = model.names[cls]
                conf  = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                original_label  = label

                # Replace "person" with face name if overlapping
                if original_label == "person" and face_detections:
                    for face in face_detections:
                        fx1, fy1, fx2, fy2 = face["box"]
                        ix1 = max(x1, fx1); iy1 = max(y1, fy1)
                        ix2 = min(x2, fx2); iy2 = min(y2, fy2)
                        if ix1 < ix2 and iy1 < iy2:
                            face_area         = (fx2 - fx1) * (fy2 - fy1)
                            intersection_area = (ix2 - ix1) * (iy2 - iy1)
                            if face_area > 0 and intersection_area / face_area > 0.3:
                                label = face["name"]
                                break

                # Direction
                cx = (x1 + x2) // 2
                if cx < w * 0.33:
                    direction = "on your left"
                elif cx > w * 0.66:
                    direction = "on your right"
                else:
                    direction = "straight ahead"

                distance_cm    = estimate_distance_cm(original_label, [x1, y1, x2, y2], w)
                distance_label = get_distance_label(distance_cm)
                distance_text  = (
                    f"{distance_cm / 100:.1f}m" if distance_cm >= 100 else f"{distance_cm}cm"
                )

                detection_obj = {
                    "label":               label,              # may be face name
                    "raw_label":           original_label,     # always YOLO class
                    "label_translated":    translate(label if label == original_label else label, lang)
                                           if label == original_label
                                           else label,         # keep person name as-is
                    "direction":           direction,
                    "direction_translated": translate(direction, lang),
                    "distance":            distance_label,
                    "distance_translated": translate(distance_label, lang),
                    "distance_cm":         distance_cm,
                    "distance_text":       distance_text,
                    "box":                 [x1, y1, x2, y2],
                    "danger":              distance_cm < 100,
                    "confidence":          round(conf, 2),
                }

                if original_label in MODES.get(mode, []):
                    visual_detections.append(detection_obj)
                audio_detections.append(detection_obj)

        has_face_danger = any(f.get("is_danger", False) for f in face_detections)

        # ── FIRE DETECTION ────────────────────────────────────
        fire_detections = detect_fire_in_frame(frame, lang)
        has_fire        = len(fire_detections) > 0
        has_danger_fire = any(f.get("danger", False) for f in fire_detections)

        # Merge fires into audio_detections so voice alerts fire
        audio_detections.extend(fire_detections)

        print(f"📊 Visual:{len(visual_detections)} Audio:{len(audio_detections)} "
              f"Faces:{len(face_detections)} Fire:{len(fire_detections)} Lang:{lang}")

        return jsonify({
            "visual_detections": visual_detections,
            "audio_detections":  audio_detections,
            "face_detections":   face_detections,
            "fire_detections":   fire_detections,
            "has_face_danger":   has_face_danger,
            "has_fire":          has_fire,
            "has_danger_fire":   has_danger_fire,
            "lang":              lang,
        })

    except Exception as e:
        print(f"Error in /detect: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/train_face', methods=['POST'])
def train_face():
    try:
        data      = request.json
        name      = data.get('name', '').strip()
        image_b64 = data.get('image')

        if not name:
            return jsonify({"success": False, "message": "Name is required"}), 400
        if not image_b64:
            return jsonify({"success": False, "message": "Image is required"}), 400

        success, message = train_face_from_image(name, image_b64)
        if success:
            return jsonify({
                "success":            True,
                "message":            message,
                "known_faces_count":  len(known_face_names),
                "known_faces":        known_face_names,
            })
        else:
            return jsonify({"success": False, "message": message}), 400

    except Exception as e:
        print(f"Error in /train_face: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500


@app.route('/list_known_faces', methods=['GET'])
def list_known_faces():
    return jsonify({"known_faces": known_face_names, "count": len(known_face_names)})


@app.route('/languages', methods=['GET'])
def languages():
    """Return available languages for the UI selector."""
    return jsonify({
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "hi", "name": "हिन्दी (Hindi)"},
        ]
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
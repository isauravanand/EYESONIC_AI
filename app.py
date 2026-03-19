from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
import face_recognition
from pathlib import Path

def safe_bgr_to_rgb(frame):
    """
    Ensures image is valid 8-bit RGB for face_recognition
    """
    if frame is None:
        return None

    # If grayscale → convert to BGR
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # If RGBA → remove alpha
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ensure correct format
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    return rgb

app = Flask(__name__)

models = {}
model_files = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt"
}

known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    
    known_faces_dir = "known_faces"
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
        print("✓ Created known_faces directory")
        return
    
    # Reset known faces
    known_face_encodings = []
    known_face_names = []
    
    try:
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(known_faces_dir, filename)
                try:
                    # Use Pillow — handles CMYK, RGBA, palette, grayscale, corrupt JPEGs
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(filepath).convert("RGB")
                    rgb_frame = np.ascontiguousarray(np.array(pil_img, dtype=np.uint8))

                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
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
        print("⚠ face_recognition library not installed. Install with: pip install face-recognition cmake dlib")
    except Exception as e:
        print(f"⚠ Error loading known faces: {e}")

def train_face_from_image(name, image_b64):
    """
    Train/register a new face encoding from a base64 image
    Returns: True if successful, error message if failed
    """
    try:
        if not name or name.strip() == "":
            return False, "Name cannot be empty"
        
        # Decode image from base64
        try:
            header, encoded = image_b64.split(",", 1)
        except:
            return False, "Invalid base64 format"

        image_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_data, np.uint8)

        frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if frame is None:
            return False, "Failed to decode image"

        # ✅ FIX 1: Handle grayscale image
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # ✅ FIX 2: Handle RGBA image (4 channels)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # ✅ FIX 3: Convert to RGB (required)
        rgb_frame = safe_bgr_to_rgb(frame)

        # ✅ FIX 4: Ensure uint8 + contiguous memory
        rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_encodings:
            return False, "No face detected in the image"

        if len(face_encodings) > 1:
            return False, f"Multiple faces detected ({len(face_encodings)}). Use single face image."

        # Save image
        known_faces_dir = "known_faces"
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)

        filename = name.strip().replace(' ', '_') + ".jpg"
        filepath = os.path.join(known_faces_dir, filename)

        cv2.imwrite(filepath, frame)

        # Reload faces
        load_known_faces()

        return True, f"✓ Face trained successfully for: {name}"

    except Exception as e:
        return False, f"Error training face: {str(e)}"
# Load known faces at startup
load_known_faces()

def load_model(model_name):
    """Load model on demand and cache it"""
    if model_name not in models:
        model_file = model_files.get(model_name, "yolov8n.pt")
        try:
            print(f"Loading model: {model_file}")
            models[model_name] = YOLO(model_file)
            print(f"✓ Model loaded: {model_name}")
        except Exception as e:
            print(f"⚠ Error loading model {model_name}: {e}")
            # Fallback to nano model
            if model_name != "yolov8n":
                print("Falling back to yolov8n")
                return load_model("yolov8n")
            return None
    return models[model_name]

# Pre-load the default model
load_model("yolov8n")

# Object size reference (approximate average dimensions in cm)
OBJECT_SIZES = {
    "person": 170,      # Average height in cm
    "chair": 80,        # Height
    "bottle": 25,       # Height
    "cup": 10,
    "bed": 200,
    "couch": 80,
    "tv": 60,
    "car": 450,         # Length
    "bus": 1000,
    "bicycle": 180,
    "dog": 60,
    "cat": 30,
    "backpack": 50,
    "handbag": 30,
    "suitcase": 70,
    "train": 2600,
    "airplane": 2400,
    "truck": 800,
    "motorcycle": 200,
    "traffic light": 300,
    "stop sign": 120,
    "remote": 15
}

def estimate_distance_cm(label, box, frame_width):
    """
    Estimate distance in centimeters based on bounding box size
    Uses focal length estimation and object size reference
    """
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Average assumed focal length (pixels) for typical webcams
    focal_length = frame_width * 0.7
    
    # Get average object height from reference (or estimate)
    object_height = OBJECT_SIZES.get(label, 30)  # Default 30cm if unknown
    
    # Calculate distance using: distance = (object_height * focal_length) / box_height
    if box_height > 0:
        distance_cm = int((object_height * focal_length) / box_height)
    else:
        distance_cm = 500  # Default fallback
    
    # Clamp to reasonable values (1cm to 50 meters)
    distance_cm = max(1, min(distance_cm, 5000))
    
    return distance_cm

def estimate_face_distance_cm(face_box, frame_width):
    """
    Estimate distance of face in centimeters
    Based on face size in the frame (smaller = farther)
    """
    left, top, right, bottom = face_box
    face_width = right - left
    face_height = bottom - top
    
    # Average face width is approximately 20cm
    average_face_width_cm = 20
    
    # Assume focal length similar to regular webcam
    focal_length = frame_width * 0.7
    
    # Distance = (real_size * focal_length) / image_size
    if face_width > 0:
        distance_cm = int((average_face_width_cm * focal_length) / face_width)
    else:
        distance_cm = 500
    
    # Clamp to reasonable values
    distance_cm = max(1, min(distance_cm, 3000))
    
    return distance_cm

def get_distance_label(distance_cm):
    """Convert distance in cm to categorical label"""
    if distance_cm < 100:
        return "very close"
    elif distance_cm < 300:
        return "near"
    else:
        return "far"

# Context-based filtering modes (for visual highlighting)
MODES = {
    "home": ["chair", "bed", "tv", "bottle", "cup", "couch", "remote", "person", "dog", "cat"],
    "outdoor": ["car", "bus", "dog", "person", "bicycle", "traffic light", "stop sign", "motorcycle", "truck"],
    "crowded": ["person", "backpack", "handbag", "car", "bicycle"],
    "travel": ["suitcase", "backpack", "handbag", "bus", "train", "car", "airplane", "person"]
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    """Check server and models status"""
    return jsonify({
        "status": "running",
        "models_loaded": list(models.keys()),
        "available_models": list(model_files.keys())
    })

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        mode = data.get('mode', 'home')
        model_name = data.get('model', 'yolov8n')  
        image_b64 = data.get('image')

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
        visual_detections = []  # For drawing boxes (mode-filtered)
        audio_detections = []   # For speaking (all detections)
        face_detections = []    # For recognized faces
        
        # ===== FACE RECOGNITION =====
        try:
            # Convert BGR to RGB for face_recognition library
            rgb_frame = safe_bgr_to_rgb(frame)
            
            # Resize frame for faster processing (optional but recommended)
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            small_frame = np.ascontiguousarray(small_frame, dtype=np.uint8)  # Fix: resize breaks contiguity
            
            # Detect faces and get encodings
            face_locations = face_recognition.face_locations(small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up face locations since we downscaled the frame
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    known_face_encodings, 
                    face_encoding, 
                    tolerance=0.6
                )
                name = "Unknown Person"
                confidence = 0
                
                # Use face distances to determine best match
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                
                # Calculate distance to face (in centimeters)
                face_distance_cm = estimate_face_distance_cm([left, top, right, bottom], w)
                
                # Convert to meters if >= 100cm
                if face_distance_cm >= 100:
                    face_distance_text = f"{face_distance_cm / 100:.1f}m"
                else:
                    face_distance_text = f"{face_distance_cm}cm"
                
                # Danger if known person is closer than 200cm (2 meters)
                is_danger = (name != "Unknown Person" and face_distance_cm < 200)
                
                face_detections.append({
                    "name": name,
                    "box": [left, top, right, bottom],
                    "confidence": round(confidence, 2),
                    "is_known": name != "Unknown Person",
                    "distance_cm": face_distance_cm,
                    "distance_text": face_distance_text,
                    "is_danger": is_danger  # True if known person within 200cm
                })
        except Exception as e:
            print(f"⚠ Face recognition error: {e}")
        
        # ===== OBJECT DETECTION (YOLO) =====
        results = model(frame, conf=0.25, verbose=False)
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                original_label = label
                
                # Check for overlap with known faces
                if original_label == "person" and face_detections:
                    for face in face_detections:
                        fx1, fy1, fx2, fy2 = face["box"]
                        # Check for intersection
                        ix1 = max(x1, fx1)
                        iy1 = max(y1, fy1)
                        ix2 = min(x2, fx2)
                        iy2 = min(y2, fy2)
                        
                        if ix1 < ix2 and iy1 < iy2:
                            face_area = (fx2 - fx1) * (fy2 - fy1)
                            intersection_area = (ix2 - ix1) * (iy2 - iy1)
                            if face_area > 0 and intersection_area / face_area > 0.3:
                                label = face["name"]
                                break
                
                # 1. Direction Logic
                cx = (x1 + x2) // 2
                if cx < w * 0.33:
                    direction = "on your left"
                elif cx > w * 0.66:
                    direction = "on your right"
                else:
                    direction = "straight ahead"
                
                # 2. Distance Estimation in Centimeters
                distance_cm = estimate_distance_cm(original_label, [x1, y1, x2, y2], w)
                distance_label = get_distance_label(distance_cm)
                
                # Convert to meters if >= 100cm
                if distance_cm >= 100:
                    distance_text = f"{distance_cm / 100:.1f}m"
                else:
                    distance_text = f"{distance_cm}cm"
                
                detection_obj = {
                    "label": label,
                    "direction": direction,
                    "distance": distance_label,
                    "distance_cm": distance_cm,
                    "distance_text": distance_text,  # For display
                    "box": [x1, y1, x2, y2],
                    "danger": (distance_cm < 100),  # Danger if closer than 1 meter
                    "confidence": round(conf, 2)
                }
                
                # Add to visual_detections if it matches the current mode
                if original_label in MODES.get(mode, []):
                    visual_detections.append(detection_obj)
                
                # Always add to audio_detections for immediate speech feedback
                audio_detections.append(detection_obj)

        print(f"📊 Detections - Visual: {len(visual_detections)}, Audio: {len(audio_detections)}, Faces: {len(face_detections)}")
        
        # Check if any known person is within danger zone (200cm)
        has_face_danger = any(face.get("is_danger", False) for face in face_detections)
        
        return jsonify({
            "visual_detections": visual_detections,  # For drawing on canvas
            "audio_detections": audio_detections,    # For speaking immediately
            "face_detections": face_detections,      # For recognized faces
            "has_face_danger": has_face_danger       # True if known person within 200cm
        })

    except Exception as e:
        print(f"Error in processing: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/train_face', methods=['POST'])
def train_face():
    """
    Train a new face with a given name
    Expects: {
        "name": "person name",
        "image": "base64 encoded image"
    }
    """
    try:
        data = request.json
        name = data.get('name', '').strip()
        image_b64 = data.get('image')
        
        if not name:
            return jsonify({"success": False, "message": "Name is required"}), 400
        
        if not image_b64:
            return jsonify({"success": False, "message": "Image is required"}), 400
        
        success, message = train_face_from_image(name, image_b64)
        
        if success:
            return jsonify({
                "success": True,
                "message": message,
                "known_faces_count": len(known_face_names),
                "known_faces": known_face_names
            })
        else:
            return jsonify({"success": False, "message": message}), 400
    
    except Exception as e:
        print(f"Error in train_face: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/list_known_faces', methods=['GET'])
def list_known_faces():
    """List all currently known faces"""
    return jsonify({
        "known_faces": known_face_names,
        "count": len(known_face_names)
    })

if __name__ == "__main__":
    # Use debug=True for automatic reload during hackathon coding
    app.run(host='0.0.0.0', port=5000, debug=True)
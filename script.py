import cv2
import face_recognition
import os
import pyttsx3
from ultralytics import YOLO
import time
from PIL import Image
import numpy as np

# 1. Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Adjust speed if needed

# 2. Load the lightweight YOLO model
print("Loading YOLO AI Model...")
model = YOLO("yolov8n.pt") 

# 3. Load known faces automatically using PIL to fix transparent PNGs
known_face_encodings = []
known_face_names = []
known_faces_dir = "known_faces"

if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)
    print(f"Created '{known_faces_dir}' folder. Please add your photos here.")

print("Loading face data...")
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(known_faces_dir, filename)
        
        try:
            # FIX: Use PIL to force the image into 8-bit RGB format, removing transparent backgrounds
            pil_image = Image.open(image_path).convert("RGB")
            rgb_img = np.array(pil_image)
            
            # Get the face encodings
            encodings = face_recognition.face_encodings(rgb_img)
            
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
                print(f" Loaded: {name}")
            else:
                print(f" Warning: No face found clearly in {filename}")
        except Exception as e:
            print(f" Error loading {filename}: {e}")

print(f"System Ready! Known people: {known_face_names}")
print("Starting EYSONIC 2.0... Press 'Q' to exit.")

# 4. Start the camera
cap = cv2.VideoCapture(0)

last_spoken = ""
last_time = 0
speech_delay = 3  # Wait 3 seconds before speaking the same thing again

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.5)
    
    detected_items = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            
            # Get YOLO bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if label == "person":
                # Only run heavy face recognition IF a person is detected
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                person_name = "Unknown person"
                
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # Lower tolerance = stricter matching (default is 0.6)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = face_distances.argmin()
                        if matches[best_match_index]:
                            person_name = known_face_names[best_match_index]
                    
                    # Draw a green box around the actual face
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, person_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    detected_items.append(person_name)
                    
                # If YOLO saw a person but couldn't find a face
                if len(face_locations) == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person (No face visible)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    detected_items.append("Person")
                    
            else:
                # Standard object detection (Chair, Car, Bottle, etc.)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                detected_items.append(label)

    # 5. Speak the results
    if detected_items:
        # Grab the first detected item to speak
        main_target = detected_items[0] 
        current_time = time.time()
        
        # Speak only if it's a new object or the 3-second delay has passed
        if main_target != last_spoken or (current_time - last_time) > speech_delay:
            engine.say(f"{main_target} detected")
            engine.runAndWait()
            last_spoken = main_target
            last_time = current_time

    # Show the video feed
    cv2.imshow("EYSONIC 2.0 AI Vision", frame)

    # Press 'Q' to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
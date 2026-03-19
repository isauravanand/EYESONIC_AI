# EYSONIC AI 👁️🔊 
**AI-Based Smart Vision Assistive Device**

EYSONIC 2.0 is an advanced hybrid assistive system designed to help visually impaired individuals navigate their surroundings. By upgrading from a purely ultrasonic-based hardware device, this system integrates real-time **Computer Vision** and **Artificial Intelligence** to not just detect obstacles, but intelligently identify objects and recognize known people, providing instant voice feedback.

## ✨ Key Features
* **Real-Time Object Detection:** Utilizes the YOLOv8 (You Only Look Once) AI model to instantly identify common objects (e.g., chairs, cars, bottles) in the user's path.
* **Facial Recognition:** Employs the `face_recognition` library to detect and identify specific people. Just drop a photo into a folder, and the system learns their face automatically.
* **Voice Feedback System:** Uses `pyttsx3` to provide auditory alerts (e.g., "Saurav detected" or "Chair detected") without freezing the video feed, thanks to background threading.
* **Hybrid Processing:** Cleans and processes complex image formats (like transparent PNGs) on the fly using Pillow (PIL) and OpenCV.
* **Lag-Free Architecture:** Optimized to run heavy AI computations on every 3rd frame, ensuring a smooth and responsive live camera feed even on standard laptop CPUs.

## 🛠️ Technology Stack
* **Language:** Python 3.10.x (Strict requirement for `dlib` compatibility on Windows)
* **Computer Vision:** OpenCV (`cv2`)
* **Object Detection AI:** Ultralytics YOLOv8 (`yolov8n.pt`)
* **Face Recognition AI:** `dlib` & `face_recognition`
* **Text-to-Speech:** `pyttsx3`
* **Data Handling:** NumPy (`< 2.0`), Pillow (PIL)

## 📁 Project Structure
```text
EYSONIC_AI/
│
├── known_faces/            # Drop images of people you want the AI to recognize here
│     ├── ayush.jpeg
│     ├── sachin.jpeg
│     └── saurav.jpg
│
├── script.py               # Main application code
├── yolov8n.pt              # YOLOv8 nano model (downloads automatically on first run)
└── README.md
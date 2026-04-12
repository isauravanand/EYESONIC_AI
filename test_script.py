import face_recognition
import cv2

image = face_recognition.load_image_file("known_faces/saurav.jpeg")
face_locations = face_recognition.face_locations(image)
print(f"Detected {len(face_locations)} faces in saurav.jpeg")

if face_locations:
    print("Face locations:", face_locations)
else:
    print("NO FACE DETECTED in the image!")

"""
Run this to find the exact dlib version issue:
python debug_dlib.py
"""
import numpy as np
from PIL import Image
import dlib
import face_recognition
import sys

print(f"Python     : {sys.version}")
print(f"dlib       : {dlib.__version__}")
print(f"numpy      : {np.__version__}")
print()

# Create a synthetic 100x100 RGB image (no file needed)
synthetic = np.zeros((100, 100, 3), dtype=np.uint8)
synthetic[:, :] = [128, 64, 32]   # some color
synthetic = np.ascontiguousarray(synthetic)

print(f"Array shape    : {synthetic.shape}")
print(f"Array dtype    : {synthetic.dtype}")
print(f"C-contiguous   : {synthetic.flags['C_CONTIGUOUS']}")
print(f"F-contiguous   : {synthetic.flags['F_CONTIGUOUS']}")
print(f"Writeable      : {synthetic.flags['WRITEABLE']}")
print()

# Test 1: dlib directly
print("Test 1: dlib.get_frontal_face_detector() on synthetic array")
try:
    detector = dlib.get_frontal_face_detector()
    result = detector(synthetic, 1)
    print(f"  ✓ dlib works directly — found {len(result)} faces (expected 0 on synthetic)")
except Exception as e:
    print(f"  ✗ dlib FAILED: {e}")
    print()
    print("  → Your dlib build is incompatible with your numpy version.")
    print("  FIX: pip uninstall dlib face-recognition -y")
    print("       pip install dlib==19.24.1 face-recognition")
    sys.exit(1)

print()

# Test 2: face_recognition on synthetic
print("Test 2: face_recognition.face_locations() on synthetic array")
try:
    locs = face_recognition.face_locations(synthetic, model="hog")
    print(f"  ✓ face_recognition works — found {len(locs)} faces")
except Exception as e:
    print(f"  ✗ face_recognition FAILED: {e}")

print()

# Test 3: real image
from pathlib import Path
known = list(Path("known_faces").glob("*.jp*g")) + list(Path("known_faces").glob("*.png"))
if known:
    test_path = known[0]
    print(f"Test 3: real image — {test_path.name}")
    pil = Image.open(test_path).convert("RGB")
    arr = np.ascontiguousarray(np.array(pil, dtype=np.uint8))
    print(f"  shape={arr.shape} dtype={arr.dtype} C={arr.flags['C_CONTIGUOUS']}")
    try:
        locs = face_recognition.face_locations(arr, model="hog")
        print(f"  ✓ Found {len(locs)} face(s)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
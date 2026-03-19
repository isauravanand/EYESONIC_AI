"""
EYSONIC Face Fix Script
Run this ONCE from your project folder:  python fix_faces.py
It will diagnose and auto-fix all images in known_faces/
"""

import os
import sys
import numpy as np

KNOWN_FACES_DIR = "known_faces"

print("=" * 55)
print("  EYSONIC Face Diagnostic & Fix Tool")
print("=" * 55)

# ── 1. Check Pillow ──────────────────────────────────────────
try:
    from PIL import Image
    print("✓ Pillow available")
except ImportError:
    print("✗ Pillow missing — run:  pip install Pillow")
    sys.exit(1)

# ── 2. Check face_recognition ────────────────────────────────
try:
    import face_recognition
    print("✓ face_recognition available")
except ImportError:
    print("✗ face_recognition missing — run:  pip install face-recognition")
    sys.exit(1)

# ── 3. Check known_faces folder ──────────────────────────────
if not os.path.exists(KNOWN_FACES_DIR):
    print(f"✗ Folder '{KNOWN_FACES_DIR}' not found — create it and add photos")
    sys.exit(1)

files = [f for f in os.listdir(KNOWN_FACES_DIR)
         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not files:
    print(f"✗ No images found in '{KNOWN_FACES_DIR}'")
    sys.exit(1)

print(f"\nFound {len(files)} image(s): {files}\n")

# ── 4. Diagnose + Fix each image ─────────────────────────────
fixed_count = 0
failed = []

for filename in files:
    path = os.path.join(KNOWN_FACES_DIR, filename)
    print(f"── {filename}")

    # Open with Pillow (most tolerant loader)
    try:
        pil_img = Image.open(path)
        print(f"   Pillow mode : {pil_img.mode}  size: {pil_img.size}")
    except Exception as e:
        print(f"   ✗ Pillow cannot open: {e}")
        failed.append(filename)
        continue

    # Convert to strict RGB
    try:
        if pil_img.mode != "RGB":
            print(f"   ⚠ Converting {pil_img.mode} → RGB")
            pil_img = pil_img.convert("RGB")

        rgb_array = np.array(pil_img, dtype=np.uint8)
        rgb_array = np.ascontiguousarray(rgb_array)

        print(f"   Array shape : {rgb_array.shape}  dtype: {rgb_array.dtype}  "
              f"C-contiguous: {rgb_array.flags['C_CONTIGUOUS']}")
    except Exception as e:
        print(f"   ✗ Conversion failed: {e}")
        failed.append(filename)
        continue

    # Try face detection
    try:
        locs = face_recognition.face_locations(rgb_array, model="hog")
        if locs:
            print(f"   ✓ {len(locs)} face(s) detected")
        else:
            print(f"   ⚠ No faces detected (photo may be blurry/small/angled)")
    except Exception as e:
        print(f"   ✗ face_recognition error: {e}")
        failed.append(filename)
        continue

    # Overwrite with clean RGB JPEG (fixes any corrupt/CMYK/RGBA issues)
    try:
        backup = path + ".bak"
        os.replace(path, backup)             # keep original as .bak
        pil_img.save(path, "JPEG", quality=95)
        print(f"   ✓ Re-saved as clean RGB JPEG  (backup: {filename}.bak)")
        fixed_count += 1
    except Exception as e:
        print(f"   ✗ Could not re-save: {e}")
        failed.append(filename)

    print()

# ── 5. Summary ───────────────────────────────────────────────
print("=" * 55)
print(f"  Fixed : {fixed_count}/{len(files)} images")
if failed:
    print(f"  Failed: {failed}")
    print("  → Replace these photos with clear, well-lit JPEGs")
else:
    print("  All images processed successfully!")
print()
print("Now restart your Flask app:  python app.py")
print("You should see:  ✓ Loaded face: <Name>")
print("=" * 55)
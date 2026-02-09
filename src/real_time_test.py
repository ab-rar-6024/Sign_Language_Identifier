import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import os

# -------------------------------
# Paths and Constants
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../models/sign_language_model.h5")
CLASS_JSON_PATH = os.path.join(BASE_DIR, "../models/class_indices.json")
IMG_WIDTH, IMG_HEIGHT = 64, 64
CONFIDENCE_THRESHOLD = 50  # Only show predictions above this %

# -------------------------------
# Load model and classes
# -------------------------------
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

# Load and safely convert JSON mapping
with open(CLASS_JSON_PATH, "r") as f:
    class_indices = json.load(f)

# Handle both {"A": 0, "B": 1, ...} and {"0": "A", "1": "B"} formats
if all(isinstance(v, int) for v in class_indices.values()):
    # Format: {"A": 0, "B": 1, ...}
    classes = {v: k for k, v in class_indices.items()}
else:
    # Format: {"0": "A", "1": "B", ...}
    classes = {int(k): v for k, v in class_indices.items()}

print(f"[INFO] Loaded {len(classes)} class mappings: {list(classes.values())[:5]}...")

# -------------------------------
# Initialize webcam
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot access webcam")

print("[INFO] Webcam started. Press 'q' to quit.")

# -------------------------------
# Real-time prediction loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect

    # Define ROI (Region of Interest)
    x0, y0, width, height = 100, 100, 300, 300
    roi = frame[y0:y0 + height, x0:x0 + width]

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.expand_dims(roi_normalized, axis=0)

    # Predict
    prediction = model.predict(roi_reshaped, verbose=0)[0]
    top_indices = prediction.argsort()[-3:][::-1]  # Top 3 predictions

    # Draw ROI box
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 2)

    # Display top 3 predictions
    for i, idx in enumerate(top_indices):
        letter = classes.get(idx, "?")
        confidence = prediction[idx] * 100
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        color = (0, 0, 255) if i == 0 else (255, 255, 255)
        y_text = y0 - 10 - i * 30
        cv2.putText(
            frame,
            f"{letter}: {confidence:.2f}%",
            (x0, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # Show main detected sign on top
    top_idx = top_indices[0]
    top_letter = classes.get(top_idx, "?")
    top_confidence = prediction[top_idx] * 100
    cv2.putText(
        frame,
        f"Detected: {top_letter} ({top_confidence:.1f}%)",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3
    )

    # Display live feed
    cv2.imshow("Sign Language Detection", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

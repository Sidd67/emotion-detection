import cv2  # type: ignore
import sys
import os
import numpy as np  # type: ignore
import time
from collections import deque
from tensorflow.keras.models import load_model  # type: ignore

# -----------------------
# CONFIGURATIONS
# -----------------------
MODEL_PATH = "emotion_model.keras"

# Safer cascade path detection
if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
    CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
else:
    CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Distance thresholds (face_area / frame_area)
MIN_FACE_AREA_RATIO = 0.04
MAX_FACE_AREA_RATIO = 0.35

# Lighting thresholds (grayscale mean brightness)
LOW_LIGHT_THRESH = 60
HIGH_LIGHT_THRESH = 200

# Padding configurations (in pixels)
FACE_PADDING = 20

# -----------------------
# LOAD RESOURCES
# -----------------------
print("Loading model and cascade...")
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print(f"Error: Cascade path invalid or file missing: {CASCADE_PATH}")
    # Fallback to local if possible
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("Error: Fallback cascade also failed. Please check haarcascade file.")

from typing import Deque, List, Tuple

# Global variables for smoothing and tracking
emotion_history: Deque[str] = deque(maxlen=10)
confidence_history: Deque[float] = deque(maxlen=10)

last_predictions: List[Tuple[str, float]] = []
frame_count: int = 0
prev_time: float = 0.0

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def check_lighting(face_gray):
    """Check if the face is properly lit using grayscale mean."""
    if face_gray is None or face_gray.size == 0:
        return "Unknown", (0, 255, 0)
    
    mean_brightness = float(np.mean(face_gray))
    if mean_brightness < LOW_LIGHT_THRESH:
        return "Low Light", (0, 0, 255)  # Red alert
    elif mean_brightness > HIGH_LIGHT_THRESH:
        return "Too Bright", (0, 0, 255)  # Red alert
    return "Lighting OK", (0, 255, 0)  # Green OK

def check_distance(face_area, frame_area):
    """Check if the face is at an optimal distance based on area ratio."""
    # Ensure no division by zero
    safe_frame_area = max(1.0, float(frame_area))
    ratio = float(face_area) / safe_frame_area
    
    if ratio < MIN_FACE_AREA_RATIO:
        return "Come Closer", (0, 0, 255)  # Red alert
    elif ratio > MAX_FACE_AREA_RATIO:
        return "Move Back", (0, 0, 255)  # Red alert
    return "Good Distance", (0, 255, 0)  # Green OK

def draw_top_panel(frame, fps, active_faces):
    """Draw a status panel at the top of the frame showing active detection count and FPS."""
    try:
        _, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (int(w), 40), (0, 0, 0), -1)
        status_text = f"Status: {active_faces} Face(s) Detected | FPS: {fps:.1f}"
        cv2.putText(frame, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    except Exception:
        pass

# -----------------------
# MAIN LOOP
# -----------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam. Check permissions or device connection.")
    sys.exit(1)

print("Starting video stream... [Press ESC to stop]")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to fetch frame from webcam.")
        break

    frame_count += 1
    fh, fw, _ = frame.shape
    frame_total_area = float(fh * fw)

    # Calculate FPS for visualization
    curr_time = time.time()
    time_diff = curr_time - prev_time
    fps = 1.0 / max(1e-5, time_diff) if prev_time > 0 else 0.0
    prev_time = curr_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    # Returns (x, y, w, h) bounding boxes
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Standardize output to an iterable list
    if not isinstance(detected_faces, np.ndarray):
        faces_list = []
    else:
        faces_list = detected_faces

    # Performance optimization: Perform prediction every 3rd frame 
    # or whenever the number of detected faces changes.
    if frame_count % 3 == 0 or len(faces_list) != len(last_predictions):
        last_predictions = []
        
        for (x, y, w, h) in faces_list:
            # Apply padding to the face crop for better context
            px1 = int(max(0, x - FACE_PADDING))
            py1 = int(max(0, y - FACE_PADDING))
            px2 = int(min(fw, x + w + FACE_PADDING))
            py2 = int(min(fh, y + h + FACE_PADDING))
            
            face_crop = frame[py1:py2, px1:px2]
            
            if face_crop.size == 0:
                last_predictions.append(("Unknown", 0.0))
                continue

            try:
                # Preprocessing: 224x224 resize + normalization [0, 1]
                face_resized = cv2.resize(face_crop, (224, 224))
                face_norm = face_resized.astype("float32") / 255.0
                face_tensor = np.expand_dims(face_norm, axis=0)

                # Inference
                preds = model.predict(face_tensor, verbose=0)[0]
                raw_conf = float(np.max(preds))
                raw_emotion = str(emotion_labels[np.argmax(preds)])
                
                # Single-face smoothing logic (Majority voting + Confidence averaging)
                if len(faces_list) == 1:
                    emotion_history.append(raw_emotion)
                    confidence_history.append(raw_conf)
                    
                    hist_list = list(emotion_history)
                    smooth_emotion = max(set(hist_list), key=hist_list.count)
                    
                    # Calculate mean confidence only for the dominant emotion in history
                    relevant_confs = [c for e, c in zip(emotion_history, confidence_history) if e == smooth_emotion]
                    smooth_conf = float(np.mean(relevant_confs)) if relevant_confs else raw_conf
                else:
                    # Multi-face detections skip smoothing to avoid identity mixups
                    smooth_emotion = raw_emotion
                    smooth_conf = raw_conf

                last_predictions.append((smooth_emotion, smooth_conf))
                
            except Exception as e:
                print(f"Inference warning: {e}")
                last_predictions.append(("Unknown", 0.0))

    # Render UI elements for detected faces
    for i, (x_raw, y_raw, w_raw, h_raw) in enumerate(faces_list):
        # Ensure we have a matching prediction
        emotion, conf = last_predictions[i] if i < len(last_predictions) else ("Unknown", 0.0)
            
        # Re-calculate padded coordinates for drawing
        x, y, w, h = int(x_raw), int(y_raw), int(w_raw), int(h_raw)
        px1 = int(max(0, x - FACE_PADDING))
        py1 = int(max(0, y - FACE_PADDING))
        px2 = int(min(fw, x + w + FACE_PADDING))
        py2 = int(min(fh, y + h + FACE_PADDING))
        
        face_roi_gray = gray[py1:py2, px1:px2]
        current_face_area = float((px2 - px1) * (py2 - py1))
        
        # Evaluate current conditions
        light_msg, light_color = check_lighting(face_roi_gray)
        dist_msg, dist_color = check_distance(current_face_area, frame_total_area)

        # UI Color Logic: Red if any alert is active, Green if all conditions are OK
        is_alert_active = (light_color == (0, 0, 255) or dist_color == (0, 0, 255))
        main_ui_color = (0, 0, 255) if is_alert_active else (0, 255, 0)

        # Draw the Bounding Box
        cv2.rectangle(frame, (px1, py1), (px2, py2), main_ui_color, 2)

        # Draw Emotion and Confidence Level
        label_str = f"{emotion} ({conf:.0%})"
        label_y_pos = int(max(20, py1 - 10))
        cv2.putText(frame, label_str, (px1, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, main_ui_color, 2)

        # Draw Contextual Alerts below the face box
        alert_y_offset = int(py2 + 25)
        cv2.putText(frame, f"Light: {light_msg}", (ST := px1, alert_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color, 1)
        alert_y_offset += 20
        cv2.putText(frame, f"Dist: {dist_msg}", (ST, alert_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 1)

    # Render the global top status panel
    draw_top_panel(frame, fps, len(faces_list))

    # Show the output frame
    cv2.imshow("Real-time Emotion Detector", frame)

    # Exit on ESC key (ASCII 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

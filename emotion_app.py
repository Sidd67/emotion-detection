import base64
import io
import os
import sys
import time
from dotenv import load_dotenv  # type: ignore
load_dotenv()  # Load .env file if present
import socket
import numpy as np  # type: ignore
import cv2  # type: ignore
import qrcode  # type: ignore
import asyncio
import urllib.request
import json
import math
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import HTMLResponse, Response, RedirectResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

# -----------------------
# GLOBAL CLOUD CONFIG
# -----------------------
# Set HF_API_KEY in your .env file or environment variables
# Example: HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxx
HF_API_KEY = os.environ.get("HF_API_KEY", "")

def query_huggingface(img_bytes):
    try:
        url = "https://api-inference.huggingface.co/models/mrm8488/vit-face-expression"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        req = urllib.request.Request(url, data=img_bytes, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=2.5) as response:
            res = json.loads(response.read().decode('utf-8'))
            if isinstance(res, list) and len(res) > 0:
                payload = res[0] if isinstance(res[0], list) else res
                full_dist = {str(item['label']).title(): float(item['score'] * 100) for item in payload}
                top = payload[0] if isinstance(payload, list) else payload
                return str(top.get('label', '')).title(), float(top.get('score', 0.0)), full_dist
    except Exception as e:
        pass
    return None

# -----------------------
# APP INITIALIZATION
# -----------------------
app = FastAPI(title="EmotionAI Real-time")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "running"}

@app.get("/lt-password")
async def get_lt_password():
    import urllib.request
    try:
        with urllib.request.urlopen("https://loca.lt/mytunnelpassword", timeout=5) as response:
            ip = response.read().decode("utf-8").strip()
            return {"ip": ip}
    except Exception as e:
        try:
            with urllib.request.urlopen("https://api.ipify.org", timeout=5) as response:
                ip = response.read().decode("utf-8").strip()
                return {"ip": ip}
        except:
            return {"ip": "Error fetching IP"}

# -----------------------
# STATIC FILES
# -----------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -----------------------
# PROJECT PARAMS (from predict.py)
# -----------------------
# Distance thresholds (face_area / frame_area)
MIN_FACE_AREA_RATIO = 0.04
MAX_FACE_AREA_RATIO = 0.35

# Lighting thresholds (grayscale mean brightness)
LOW_LIGHT_THRESH = 60
HIGH_LIGHT_THRESH = 200

# Padding configurations (in pixels)
# 🎯 INCREASED to 40 to ensure Eyebrows & Forehead are fully visible for Angry/Surprise
FACE_PADDING = 40

def check_lighting(face_gray):
    """Check if the face is properly lit using grayscale mean."""
    if face_gray is None or face_gray.size == 0:
        return "Unknown", (0, 255, 0)
    
    mean_brightness = float(np.mean(face_gray))
    if mean_brightness < LOW_LIGHT_THRESH:
        return "Low Light", (0, 0, 255)
    elif mean_brightness > HIGH_LIGHT_THRESH:
        return "Too Bright", (0, 0, 255)
    return "Lighting OK", (0, 255, 0)

def check_distance(face_area, frame_area):
    """Check if the face is at an optimal distance based on area ratio."""
    safe_frame_area = max(1.0, float(frame_area))
    ratio = float(face_area) / safe_frame_area
    if ratio < MIN_FACE_AREA_RATIO:
        return "Come Closer", (0, 0, 255)
    elif ratio > MAX_FACE_AREA_RATIO:
        return "Move Back", (0, 0, 255)
    return "Good Distance", (0, 255, 0)

def draw_top_panel(frame, fps, active_faces):
    """Draw a status panel at the top of the frame showing active detection count and FPS."""
    try:
        _, w, _ = frame.shape
        scale = w / 640.0
        cv2.rectangle(frame, (0, 0), (int(w), int(40 * scale)), (0, 0, 0), -1)
        status_text = f"Status: {active_faces} Face(s) Detected | FPS: {fps:.1f}"
        cv2.putText(frame, status_text, (int(15 * scale), int(28 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (255, 255, 255), max(1, int(2 * scale)))
    except Exception:
        pass

# -----------------------
# FACE DETECTOR
# -----------------------
if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
    CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
else:
    CASCADE_PATH = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print(f"Error: Cascade path invalid or file missing: {CASCADE_PATH}")

# -----------------------
# EMOTION MODEL (mini-xception)
# -----------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_mini_xception.h5")
labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

print("Loading High-Accuracy Xception emotion model for server...")
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    emotion_model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    emotion_model = None

# -----------------------
# PREPROCESS
# -----------------------
def preprocess_face(face):
    """Xception Model expects 64x64 Grayscale: (None, 64, 64, 1)"""
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # 🎯 ACCURACY BOOST: Apply CLAHE to enhance facial contrast instantly
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_face = clahe.apply(gray_face)
    
    resized_face = cv2.resize(gray_face, (64, 64))
    norm_face = resized_face.astype("float32") / 255.0
    face_tensor = np.expand_dims(norm_face, axis=-1)  # Add channel dimension (64, 64, 1)
    face_tensor = np.expand_dims(face_tensor, axis=0) # Add batch dimension (1, 64, 64, 1)
    return face_tensor

# -----------------------
# PREDICTION
# -----------------------
def predict_emotion(face_tensor):
    if emotion_model is None:
        return "Unknown", 0.0, {}
    preds = emotion_model.predict(face_tensor, verbose=0)[0]
    conf = float(np.max(preds))
    emotion = str(labels[np.argmax(preds)])
    
    # Generate full distribution for the floating UI
    dist = {str(labels[i]): float(preds[i] * 100) for i in range(len(labels))}
    return emotion, conf, dist

# -----------------------
# UTILITY: GET LOCAL IP
# -----------------------
def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# -----------------------
# ROUTES
# -----------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    path = os.path.join(STATIC_DIR, "index.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/pc", response_class=HTMLResponse)
async def pc():
    path = os.path.join(STATIC_DIR, "pc.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/mobile")
async def mobile(request: Request):
    user_agent = request.headers.get("user-agent", "").lower()
    
    # Restrict PC users from accessing the mobile link
    is_mobile = "mobi" in user_agent or "android" in user_agent or "iphone" in user_agent
    if not is_mobile:
        return RedirectResponse(url="/pc")

    path = os.path.join(STATIC_DIR, "mobile.html")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/face-cam", response_class=HTMLResponse)
async def face_cam():
    path = os.path.join(STATIC_DIR, "face_camera.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/status")
async def status():
    return {"status": "ok", "model": "emotion_model.keras", "ip": get_local_ip()}

@app.get("/qr")
async def generate_qr():
    ip = get_local_ip()
    url = f"http://{ip}:8000/mobile"
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")

# -----------------------
# WEBSOCKET
# -----------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client: str = "mobile"):
    await websocket.accept()
    print(f"[+] Client connected ({client})")

    frame_count: int = 0
    emotion_buffer: "deque[str]" = deque(maxlen=3)
    confidence_buffer: "deque[float]" = deque(maxlen=3)
    
    last_emotion = "Scanning..."
    last_conf = 0.0
    last_dist = {}
    last_cloud_ping: float = 0.0
    
    prev_time = time.time()

    try:
        while True:
            data = await websocket.receive_json()

            frame_b64 = data.get("frame")
            
            if not frame_b64:
                continue

            img_bytes = base64.b64decode(frame_b64)
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                continue
            
            # Update timers for FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            fps = 1.0 / max(1e-5, time_diff) if prev_time > 0 else 0.0
            prev_time = curr_time

            fh, fw = frame.shape[:2]
            frame_total_area = float(fh * fw)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.2, 3)
            
            if not isinstance(detected_faces, np.ndarray):
                faces = []
            else:
                faces = detected_faces

            frame_count += 1
            response: dict = {}

            if len(faces) == 0:
                response = {
                    "status": "error",
                    "type": "no_face",
                    "message": "⚠️ No face detected",
                }
                emotion_buffer.clear()
                confidence_buffer.clear()

            elif len(faces) > 1:
                response = {
                    "status": "error",
                    "type": "multiple_faces",
                    "message": "Multiple faces detected. Only one person allowed.",
                }
                emotion_buffer.clear()
                confidence_buffer.clear()

            else:
                x_val, y_val, w_val, h_val = faces[0]
                x, y, w, h = int(x_val), int(y_val), int(w_val), int(h_val)
                
                # App padding
                px1 = int(max(0, x - FACE_PADDING))
                py1 = int(max(0, y - FACE_PADDING))
                px2 = int(min(fw, x + w + FACE_PADDING))
                py2 = int(min(fh, y + h + FACE_PADDING))
                
                face_crop = frame[py1:py2, px1:px2]
                face_roi_gray = gray[py1:py2, px1:px2]
                current_face_area = max(1, (px2 - px1) * (py2 - py1))
                
                # Context evaluations
                light_msg, light_color = check_lighting(face_roi_gray)
                dist_msg, dist_color = check_distance(current_face_area, frame_total_area)

                is_alert_active = (light_color == (0, 0, 255) or dist_color == (0, 0, 255))
                
                # ⚡ ACCURACY/SPEED BOOST: predict every 2nd frame (instead of 3rd) for faster reactivity
                if frame_count % 2 == 0 and face_crop.size > 0:
                    try:
                        # LOCAL PREDICTION
                        ft = preprocess_face(face_crop)
                        emotion, conf, dist = predict_emotion(ft)
                        last_dist = dist  # Keep UI updated instantly
                        
                        emotion_buffer.append(emotion)
                        confidence_buffer.append(conf)
                        
                        # CLOUD OVERRIDE (Fires asynchronously every 1.5 seconds)
                        if curr_time - last_cloud_ping > 1.5:
                            last_cloud_ping = curr_time
                            _, buffer = cv2.imencode('.jpg', face_crop)
                            b_data = buffer.tobytes()
                            
                            async def fetch_cloud(b):
                                loop = asyncio.get_running_loop()
                                res = await loop.run_in_executor(None, query_huggingface, b)
                                if res:
                                    cloud_emot, cloud_conf, cloud_dist = res
                                    # Override local buffer with SOTA cloud prediction
                                    
                                    global last_dist
                                    last_dist = cloud_dist
                                    
                                    emotion_buffer.clear()
                                    confidence_buffer.clear()
                                    for _ in range(4): 
                                        emotion_buffer.append(cloud_emot)
                                        confidence_buffer.append(cloud_conf)
                                    print(f"☁️ Cloud Vision Override: {cloud_emot} ({cloud_conf:.1%})")

                            asyncio.create_task(fetch_cloud(b_data))
                        
                        hist_list = list(emotion_buffer)
                        smooth_emotion = max(set(hist_list), key=hist_list.count)
                        
                        # Avg conf
                        relevant_confs = [c for e, c in zip(emotion_buffer, confidence_buffer) if e == smooth_emotion]
                        smooth_conf = float(np.mean(relevant_confs)) if relevant_confs else conf
                        
                        last_emotion = smooth_emotion
                        last_conf = smooth_conf
                    except Exception as e:
                        print("Inference warning:", e)
                        
                response = {
                    "status": "success",
                    "message": "Face detected successfully.",
                    "emotion": last_emotion,
                    "confidence": float(f"{float(last_conf) * 100:.1f}"),
                    "distribution": last_dist,
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                }
                        
                if client == "pc":
                    try:
                        scale = fw / 640.0
                        thick = max(1, int(2 * scale))
                        main_ui_color = (0, 0, 255) if is_alert_active else (0, 255, 0)
                        cv2.rectangle(frame, (px1, py1), (px2, py2), main_ui_color, thick)
                        label_str = f"{last_emotion} ({last_conf:.0%})"
                        label_y_pos = int(max(20 * scale, py1 - 10 * scale))
                        cv2.putText(frame, label_str, (px1, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, main_ui_color, thick)
                        alert_y_offset = int(py2 + 25 * scale)
                        cv2.putText(frame, f"Light: {light_msg}", (px1, alert_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, light_color, max(1, thick))
                        alert_y_offset += int(20 * scale)
                        cv2.putText(frame, f"Dist: {dist_msg}", (px1, alert_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, dist_color, max(1, thick))
                    except Exception as e:
                        print("PC UI drawing error:", e)

            if client == "pc":
                try:
                    draw_top_panel(frame, fps, len(faces) if 'faces' in locals() else 0)
                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    response["frame"] = base64.b64encode(buffer).decode()  # type: ignore
                except Exception as e:
                    print("PC UI encoding error:", e)

            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("[-] Client disconnected")
    except Exception as e:
        print("[!] Error:", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
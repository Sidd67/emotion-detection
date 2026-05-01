# 🎭 EmotionAI — Real-Time Emotion Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A full-stack, real-time facial emotion recognition system powered by deep learning.**  
Detects 7 emotions live from your webcam — on PC or mobile — via WebSockets.

</div>

---

## ✨ Features

- 🎯 **7-Class Emotion Detection** — Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- ⚡ **Real-Time WebSocket Streaming** — Ultra-low latency face analysis at 30 FPS
- 📱 **Multi-Device Support** — Separate optimized UIs for PC and Mobile
- ☁️ **Hybrid Inference** — Local mini-Xception model + Cloud HuggingFace Vision API override
- 💡 **Smart Lighting & Distance Checks** — Guides user for best camera conditions
- 📊 **Live Confidence Distribution** — Floating bar chart showing all 7 emotion probabilities
- 📸 **QR Code Access** — Auto-generates QR code for instant mobile access on LAN
- 🖥️ **Desktop Annotated Feed** — Bounding boxes + labels rendered server-side for PC mode

---

## 🧠 Model Architecture

| Component | Details |
|-----------|---------|
| **Local Model** | Mini-Xception (64×64 Grayscale) — `emotion_mini_xception.h5` |
| **Preprocessing** | CLAHE contrast enhancement + normalization |
| **Smoothing** | Temporal buffer (last 3 frames) for stable predictions |
| **Cloud Fallback** | HuggingFace `mrm8488/vit-face-expression` (ViT) every 1.5s |
| **Face Detector** | Haar Cascade (`haarcascade_frontalface_default.xml`) |

---

## 📁 Project Structure

```
EmotionAI/
├── emotion_app.py              # 🚀 FastAPI backend (WebSocket server + routes)
├── predict.py                  # 🔍 Standalone prediction script
├── train_model.py              # 🏋️ Model training script
├── requirements.txt            # 📦 Python dependencies
├── emotion_mini_xception.h5    # 🧠 Trained local model (mini-Xception)
├── haarcascade_frontalface_default.xml  # 👤 Face detector
├── static/
│   ├── index.html              # 🏠 Landing page (device selector)
│   ├── pc.html                 # 🖥️ PC dashboard UI
│   ├── mobile.html             # 📱 Mobile camera UI
│   ├── face_camera.html        # 📷 Face camera view
│   └── desktop.html            # 🖥️ Desktop alternate UI
├── train/                      # 📂 Training dataset (FER-2013 format)
└── test/                       # 📂 Test dataset
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Sidd67/EmotionAI.git
cd EmotionAI
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Server

```bash
uvicorn emotion_app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open in Browser

| Mode | URL |
|------|-----|
| 🏠 Home / Device Selector | http://localhost:8000 |
| 🖥️ PC Dashboard | http://localhost:8000/pc |
| 📱 Mobile (LAN) | http://\<your-ip\>:8000/mobile |
| 📷 Face Camera | http://localhost:8000/face-cam |
| 📊 QR Code for Mobile | http://localhost:8000/qr |

---

## 📱 Mobile Access (LAN)

1. Make sure PC and mobile are on the **same Wi-Fi network**
2. Open `http://localhost:8000/qr` on your PC
3. **Scan the QR code** with your mobile camera
4. Mobile browser opens the optimized mobile interface automatically

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/pc` | GET | PC dashboard |
| `/mobile` | GET | Mobile UI (redirects PC to /pc) |
| `/face-cam` | GET | Face camera view |
| `/qr` | GET | QR code PNG for mobile LAN access |
| `/health` | GET | Health check — `{"status": "running"}` |
| `/api/status` | GET | Server status + model info + IP |
| `/lt-password` | GET | LocalTunnel password helper |
| `/ws?client=pc` | WebSocket | Real-time emotion stream (PC mode) |
| `/ws?client=mobile` | WebSocket | Real-time emotion stream (Mobile mode) |

### WebSocket Message Format

**Send (Client → Server):**
```json
{
  "frame": "<base64-encoded-JPEG>"
}
```

**Receive (Server → Client) — Success:**
```json
{
  "status": "success",
  "emotion": "Happy",
  "confidence": 94.5,
  "distribution": {
    "Angry": 1.2, "Disgust": 0.5, "Fear": 0.8,
    "Happy": 94.5, "Sad": 1.0, "Surprise": 1.5, "Neutral": 0.5
  },
  "bbox": {"x": 120, "y": 80, "w": 200, "h": 200}
}
```

**Receive (Server → Client) — Error:**
```json
{
  "status": "error",
  "type": "no_face",
  "message": "⚠️ No face detected"
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI (Python) |
| **WebSockets** | FastAPI WebSocket |
| **ML Framework** | TensorFlow / Keras |
| **Face Detection** | OpenCV Haar Cascade |
| **Cloud AI** | HuggingFace Inference API |
| **Frontend** | Vanilla HTML + CSS + JavaScript |
| **QR Code** | `qrcode` Python library |
| **Tunneling** | LocalTunnel (`lt`) |

---

## 🎓 Dataset

This model was trained on the **FER-2013** dataset:
- 35,887 grayscale images (48×48)
- 7 emotion classes
- Resized to 64×64 for mini-Xception input

> Place training data in `train/` and `test/` folders in FER-2013 format:
> ```
> train/
>   Angry/   Disgust/  Fear/  Happy/
>   Sad/     Surprise/ Neutral/
> ```

---

## ⚙️ Configuration

Key thresholds in `emotion_app.py`:

```python
MIN_FACE_AREA_RATIO = 0.04   # Minimum face size (too far away)
MAX_FACE_AREA_RATIO = 0.35   # Maximum face size (too close)
LOW_LIGHT_THRESH    = 60     # Grayscale mean below = "Low Light"
HIGH_LIGHT_THRESH   = 200    # Grayscale mean above = "Too Bright"
FACE_PADDING        = 40     # Pixels added around face crop
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Author

**Siddharth** — [@Sidd67](https://github.com/Sidd67)

> ⭐ If you find this project useful, please give it a star!

# 🛡️ DrowseGuard — Driver Drowsiness Detection

**TensorFlow (MobileNetV2) + MediaPipe FaceLandmarker + OpenCV | Real-Time**

Detects driver state in real time from a webcam feed:

| State | Condition | Trigger |
|-------|-----------|---------|
| 😊 **Awake** | Eyes open, alert | — |
| 😴 **Eyes Closed** | EAR below threshold | Alarm after ~2.5s of continuous closure |
| 🥱 **Yawning** | Wide mouth opening (MAR) or CNN-confirmed | Logged, no alarm |

Two interchangeable front ends share the same detection pipeline:
- **`app.py`** — a Streamlit web dashboard with a live MJPEG video stream
- **`detect.py`** — a lightweight OpenCV desktop app with a 3-tier audio alarm

---

## 📁 Project Structure

```
Drowsiness_detection/
├── app.py                     # Streamlit web app (entry point)
├── detect.py                  # Desktop OpenCV app (entry point)
├── face_landmarker.task       # MediaPipe face landmark model (478 points)
├── requirements.txt
├── packages.txt                # apt packages needed for Streamlit Cloud deployment
├── .streamlit/
│   └── config.toml             # Streamlit theme
├── models/
│   ├── __init__.py
│   └── drowsiness_model_ft.keras   # pretrained MobileNetV2 classifier (awake/yawning)
├── utils/
│   ├── __init__.py
│   ├── face_utils.py           # MediaPipe landmarks, EAR/MAR math, ROI extraction
│   └── state_tracker.py        # detect.py's temporal smoothing + 3-tier pygame alarm
└── audio/
    ├── normal_alarm.wav         # used by detect.py's 3-tier alarm (see below)
    ├── short_alarm.mp3
    └── power_alarm.wav
```

> The `models/drowsiness_model_ft.keras` checkpoint is a pretrained artifact bundled
> with this repo — there is no training script here. It was fine-tuned on the
> [Kaggle "drowsiness-dataset"](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)
> (`dheerajperumandla/drowsiness-dataset`), reduced to a 2-class awake vs. yawning
> face-image setup. If you want to retrain or fine-tune further, you'll need to
> rebuild a training pipeline against that dataset.

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
`face_landmarker.task` and `models/drowsiness_model_ft.keras` are already included in
the repo — no separate downloads needed.

### 2a. Run the web app
```bash
streamlit run app.py
```
Opens a browser dashboard. Click **▶ Start** in the sidebar to open the webcam and
begin detection; **⏹ Stop** releases it.

### 2b. Run the desktop app
```bash
python detect.py                         # webcam
python detect.py --source video.mp4      # video file instead of a webcam
python detect.py --threshold 3.0         # custom alarm delay (default 2.5s)
python detect.py --no_landmarks          # hide the landmark overlay
```

**Keyboard controls (desktop app only):**
| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `R` | Reset alarm & timer |
| `S` | Save a screenshot to `screenshots/` |

---

## 🧠 How It Works

```
Webcam frame
     │
     ▼
MediaPipe FaceLandmarker  ──►  478 face landmarks (468 face + 10 iris)
     │
     ├──► EAR from 6 eye points   ──► EAR < 0.20            ──► EYES CLOSED
     ├──► MAR from 12 lip points  ──► MAR > 0.55             ──► YAWNING
     └──► otherwise: crop face ROI ──► MobileNetV2 CNN       ──► YAWNING vs AWAKE
                                        (only accepted if confidence > 60%)
                                                │
                                8-frame majority-vote smoothing
                                                │
                              Eyes closed continuously ≥ 2.5s ──► 🔔 ALARM
```

**Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** are geometric ratios
computed directly from landmark coordinates — cheap, reliable, and don't need the
CNN. The CNN only runs on ambiguous frames (eyes open, mouth not obviously wide),
and is throttled to every 3rd such frame in the web app to keep it fast.

**Alarm timer robustness:** a single missed detection or borderline EAR reading
doesn't reset the drowsy countdown — the timer only resets if eyes have looked
genuinely open for more than a short grace period (0.5s). This prevents brief
MediaPipe hiccups from indefinitely delaying the alarm.

---

## 🌐 Web app (`app.py`) architecture

The webcam is opened directly with OpenCV on the machine running Streamlit (no
browser camera permissions, no WebRTC/ICE negotiation). Capture and detection run
continuously in a background thread, independent of Streamlit's UI refresh cycle,
and are served to the browser as a plain **MJPEG stream** over a small local HTTP
server — the browser renders it natively, like a live IP camera feed. This keeps
video smooth (~20-30 fps depending on hardware) and avoids the flicker/lag that
comes from redrawing video through Streamlit's own rerun cycle. Only the small
text metrics (EAR/MAR/state/FPS) are updated via a lightweight Streamlit fragment.

The alarm tone is synthesized in-browser via the Web Audio API (no audio files
needed for the web app — those in `audio/` are only used by `detect.py`).

## 🖥️ Desktop app (`detect.py`) alarm tiers

Unlike the web app's single alarm state, `detect.py` escalates through 3 levels
via `utils/state_tracker.py`, using `pygame` for audio (falls back to a generated
beep tone if a file in `audio/` is missing):

| Level | Trigger | Sound |
|-------|---------|-------|
| Normal | Isolated closure ≥ alarm threshold (default 2.5s) | `audio/normal_alarm.wav` |
| Short | ≥2 alarms within the last 60s (repeated microsleeps) | `audio/short_alarm.mp3` |
| Power | Single continuous closure ≥ 5s (severely drowsy) | `audio/power_alarm.wav` (loops) |

---

## 🔧 Configuration

Key thresholds are duplicated near the top of both `app.py` and `detect.py`
(kept in sync manually since they're two separate entry points):

| Constant | Default | Effect |
|----------|---------|--------|
| `EAR_THRESHOLD` | 0.20 | Below this → eyes considered closed |
| `MAR_THRESHOLD` | 0.55 | Above this → mouth considered a yawn |
| `CONF_THRESHOLD` | 0.60 | Minimum CNN confidence to accept a YAWNING call on ambiguous frames |
| `ALARM_SECS` / `--threshold` | 2.5s | Continuous eye closure before the alarm fires |
| `DROWSY_GRACE_SECS` (`app.py` only) | 0.5s | Tolerance for brief detection blips before resetting the alarm timer |
| `CNN_EVERY_N_FRAMES` (`app.py` only) | 3 | Only run the CNN on every Nth ambiguous frame |
| `POWER_SECS` (`detect.py` only) | 5.0s | Continuous closure before escalating to the "power" alarm |

---

## 🎯 Model

MobileNetV2, fine-tuned to classify a cropped face ROI as **awake** vs. **yawning**
(eye-closed detection is handled entirely by the EAR geometric rule, not the CNN).
The `.keras` checkpoint in `models/` is loaded directly by both apps at startup —
no training step is required to use this repo as-is.

---

## 📦 Tech Stack

Streamlit · TensorFlow/Keras · MediaPipe FaceLandmarker · OpenCV · pygame (desktop
audio) · NumPy

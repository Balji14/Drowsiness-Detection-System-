"""
detect.py  —  MediaPipe edition
Real-time Driver Drowsiness Detection.

Detection strategy (hybrid):
  - CLOSED  : EAR < 0.20  (geometric rule — no CNN needed)
  - YAWNING : MobileNetV2 CNN on full face ROI (trained on yawn/no_yawn face images)
  - AWAKE   : neither of the above

Usage:
    python detect.py                              # webcam
    python detect.py --source video.mp4          # video file
    python detect.py --threshold 3.0             # custom alarm delay

Controls:
    Q / ESC  →  quit
    R        →  reset alarm & timer
    S        →  save screenshot

Download the MediaPipe model first:
    wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import cv2
import numpy as np
import argparse
import time
import os
import tensorflow as tf

from utils.face_utils import (
    FaceAnalyzer,
    RIGHT_EAR_PTS, LEFT_EAR_PTS
)
from utils.state_tracker import (
    DrowsinessTracker, STATE_NAMES, STATE_COLORS_BGR,
    STATE_AWAKE, STATE_CLOSED, STATE_YAWN
)

# ─── Config ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = 'models/drowsiness_model_ft.keras'
DEFAULT_LANDMARKER = 'face_landmarker.task'
EAR_THRESHOLD      = 0.20   # below this → CLOSED (rule-based, reliable)
MAR_THRESHOLD      = 0.55   # above this → YAWNING (rule-based, reliable)
CONF_THRESHOLD     = 0.60   # CNN must be this confident to call YAWNING
DISPLAY_SIZE       = (960, 540)


def parse_args():
    p = argparse.ArgumentParser(description='Real-time Drowsiness Detection (MediaPipe)')
    p.add_argument('--source',       default='0')
    p.add_argument('--model',        default=DEFAULT_MODEL)
    p.add_argument('--landmarker',   default=DEFAULT_LANDMARKER)
    p.add_argument('--alarm',        default=None)
    p.add_argument('--threshold',    type=float, default=2.5)
    p.add_argument('--no_landmarks', action='store_true')
    return p.parse_args()


def preprocess_roi(roi, size):
    """Resize and normalise a BGR crop for CNN inference."""
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (size, size))
    roi = roi.astype(np.float32) / 255.0
    return np.expand_dims(roi, axis=0)


def draw_hud(frame, state, alarm_active, seconds_drowsy,
             fps, confidence, face_found, ear, mar):
    h, w = frame.shape[:2]
    color = STATE_COLORS_BGR.get(state, (200, 200, 200))
    label = STATE_NAMES.get(state, 'UNKNOWN')

    # Semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (360, 145), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, label, (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {confidence*100:.0f}%  |  FPS: {fps:.1f}",
                (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"EAR: {ear:.2f}  |  MAR: {mar:.2f}",
                (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
    cv2.putText(frame, "Q=quit  R=reset  S=screenshot",
                (20, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)

    # Drowsy progress bar
    if seconds_drowsy > 0:
        bar_w = w - 40
        fill  = min(seconds_drowsy / 2.5, 1.0)
        bar_c = (0, 0, 255) if fill >= 1.0 else (0, 165, 255)
        cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 15), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h - 30),
                      (20 + int(bar_w * fill), h - 15), bar_c, -1)
        cv2.putText(frame, f"Drowsy: {seconds_drowsy:.1f}s",
                    (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_c, 1)

    # Blinking alarm banner
    if alarm_active and int(time.time() * 3) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 6)
        cv2.putText(frame, 'WAKE UP!',
                    (w // 2 - 110, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)

    if not face_found:
        cv2.putText(frame, 'No face detected', (w // 2 - 120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)


def main():
    args = parse_args()

    print(f"Loading model : {args.model}")
    model = tf.keras.models.load_model(args.model)
    # Read the input size the model was actually trained with
    model_size = model.input_shape[1]   # (None, H, W, 3) → H
    print(f"Model loaded  ✓  (input {model_size}×{model_size})")

    print(f"Loading MediaPipe FaceLandmarker: {args.landmarker}")
    analyzer = FaceAnalyzer(model_path=args.landmarker)
    print("FaceLandmarker ready ✓")

    tracker = DrowsinessTracker(
        alarm_sound_path=args.alarm,
        alarm_threshold=args.threshold
    )

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.source}")

    fps_timer = time.time()
    fps       = 0.0
    frame_idx = 0
    ear = mar = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - fps_timer, 1e-9)
        fps_timer = now

        # ── MediaPipe face landmark detection ────────────────────────────────
        result     = analyzer.detect(frame)
        coords     = analyzer.get_pixel_coords(result, frame.shape)
        face_found = coords is not None

        raw_state  = STATE_AWAKE
        confidence = 1.0

        if face_found:
            if not args.no_landmarks:
                analyzer.draw_landmarks(frame, coords)

            ear = (analyzer.eye_aspect_ratio(coords, RIGHT_EAR_PTS) +
                   analyzer.eye_aspect_ratio(coords, LEFT_EAR_PTS)) / 2.0
            mar = analyzer.mouth_aspect_ratio(coords)

            # ── Rule 1: EAR → CLOSED (reliable geometric signal) ─────────────
            if ear < EAR_THRESHOLD:
                raw_state  = STATE_CLOSED
                confidence = 1.0

            # ── Rule 2: MAR → YAWNING (reliable geometric signal) ────────────
            elif mar >= MAR_THRESHOLD:
                raw_state  = STATE_YAWN
                confidence = 1.0

            # ── Rule 3: CNN on face ROI → AWAKE vs subtle YAWNING ────────────
            else:
                fx, fy, fw, fh = analyzer.get_face_bbox(coords)
                hf, wf = frame.shape[:2]
                pad = 20
                face_roi = frame[max(0, fy - pad):min(hf, fy + fh + pad),
                                 max(0, fx - pad):min(wf, fx + fw + pad)]

                if face_roi.size > 0:
                    pred = model.predict(
                        preprocess_roi(face_roi, model_size), verbose=0)[0]
                    # Model output (empirically verified):
                    #   index 0 → yawning probability
                    #   index 1 → awake probability
                    yawn_conf  = float(pred[0])
                    awake_conf = float(pred[1])

                    if yawn_conf >= CONF_THRESHOLD:
                        raw_state  = STATE_YAWN
                        confidence = yawn_conf
                    else:
                        raw_state  = STATE_AWAKE
                        confidence = awake_conf

        smoothed, alarm_triggered, seconds_drowsy = tracker.update(raw_state)

        draw_hud(frame, smoothed, alarm_triggered,
                 seconds_drowsy, fps, confidence,
                 face_found, ear, mar)

        display = cv2.resize(frame, DISPLAY_SIZE)
        cv2.imshow('Driver Drowsiness Detection', display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            tracker.reset()
            print("[Reset] Cleared.")
        elif key == ord('s'):
            os.makedirs('screenshots', exist_ok=True)
            path = f'screenshots/shot_{frame_idx:05d}.jpg'
            cv2.imwrite(path, frame)
            print(f"Screenshot → {path}")

    cap.release()
    cv2.destroyAllWindows()

    stats = tracker.get_stats()
    print(f"\n{'='*40}\nSession Summary")
    print(f"Frames : {stats['frames_processed']}")
    print(f"Alarms : {stats['alarms_triggered']}")
    for s, pct in stats['state_percentages'].items():
        print(f"  {s}: {pct}%")


if __name__ == '__main__':
    main()

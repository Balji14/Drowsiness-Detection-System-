"""
utils/face_utils.py  —  MediaPipe 0.10+ rewrite
Face landmark detection and ROI extraction using MediaPipe FaceLandmarker.

No dlib, no .dat file, no C++ compiler required.
Install: pip install opencv-python mediapipe

MediaPipe FaceLandmarker gives 478 landmarks (468 face + 10 iris).
Landmark indices used here (from FaceLandmarksConnections):
  Right eye (camera-left): 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163
  Left eye  (camera-right): 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390
  Lips: 0, 13, 14, 17, 37, 39, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
        146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312,
        314, 317, 318, 321, 324, 375, 402, 405, 409, 415
Download model:
  wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── Landmark index groups (MediaPipe 478-point model) ────────────────────────
RIGHT_EYE_IDX = [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
LEFT_EYE_IDX  = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]

# EAR-specific 6 points per eye (vertical + horizontal pairs)
RIGHT_EAR_PTS = [33, 160, 158, 133, 153, 144]   # p1..p6
LEFT_EAR_PTS  = [263, 385, 387, 362, 380, 374]

# Mouth: outer lip contour points for MAR
MOUTH_OUTER_IDX = [61, 291, 39, 269, 0, 17, 405, 321, 314, 84, 178, 402]
MOUTH_ALL_IDX   = [
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
    146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312,
    314, 317, 318, 321, 324, 375, 402, 405, 409, 415
]

# Drawing colors (BGR)
COLOR_EYE   = (0, 255, 255)
COLOR_MOUTH = (255, 0, 255)
COLOR_FACE  = (0, 255, 0)


class FaceAnalyzer:
    def __init__(self,
                 model_path='face_landmarker.task',
                 min_face_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Args:
            model_path: path to face_landmarker.task (downloaded separately)
            min_face_confidence: detection threshold
            min_tracking_confidence: tracking threshold
        """
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=min_face_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp_vision.RunningMode.IMAGE
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        """
        Run MediaPipe on a BGR frame.
        Returns FaceLandmarkerResult (landmarks in normalized [0,1] coords).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect(mp_image)

    def get_pixel_coords(self, result, frame_shape, face_idx=0):
        """
        Convert normalized landmark coords → pixel (x, y) numpy array.
        Returns array of shape (478, 2) or None if no face detected.
        """
        if not result.face_landmarks or face_idx >= len(result.face_landmarks):
            return None
        h, w = frame_shape[:2]
        lm = result.face_landmarks[face_idx]
        coords = np.array([[int(p.x * w), int(p.y * h)] for p in lm],
                          dtype=np.int32)
        return coords

    # ── EAR / MAR ─────────────────────────────────────────────────────────────
    def eye_aspect_ratio(self, coords, ear_pts):
        """
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        ear_pts: list of 6 landmark indices [p1, p2, p3, p4, p5, p6]
        Threshold: EAR < 0.20 → likely closed
        """
        p = [coords[i] for i in ear_pts]
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        return (A + B) / (2.0 * C + 1e-6)

    def mouth_aspect_ratio(self, coords):
        """
        MAR = vertical opening / horizontal width.
        Threshold: MAR > 0.6 → likely yawning.
        """
        pts = coords[MOUTH_OUTER_IDX]
        # Vertical: top-bottom pairs
        A = np.linalg.norm(pts[4] - pts[5])   # top-bottom center
        B = np.linalg.norm(pts[6] - pts[7])   # upper-lower inner
        # Horizontal: left-right corners
        C = np.linalg.norm(pts[0] - pts[1])
        return (A + B) / (2.0 * C + 1e-6)

    # ── ROI extraction ────────────────────────────────────────────────────────
    def extract_roi(self, frame, coords, indices, padding=8, target=(64, 64)):
        """
        Crop a bounding box around the given landmark indices.
        Returns resized ROI or None if the crop is invalid.
        """
        pts = coords[indices]
        x, y, w, h = cv2.boundingRect(pts)
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_frame, x + w + padding)
        y2 = min(h_frame, y + h + padding)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
            return None
        return cv2.resize(roi, target)

    def extract_left_eye_roi(self, frame, coords, **kwargs):
        return self.extract_roi(frame, coords, LEFT_EYE_IDX, **kwargs)

    def extract_right_eye_roi(self, frame, coords, **kwargs):
        return self.extract_roi(frame, coords, RIGHT_EYE_IDX, **kwargs)

    def extract_mouth_roi(self, frame, coords, **kwargs):
        return self.extract_roi(frame, coords, MOUTH_ALL_IDX, **kwargs)

    # ── Drawing ───────────────────────────────────────────────────────────────
    def draw_landmarks(self, frame, coords, draw_mesh=False):
        """Draw eye and mouth contours on the frame."""
        if coords is None:
            return

        # Eye contours
        for idx_set in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
            pts = coords[idx_set]
            hull = cv2.convexHull(pts)
            cv2.drawContours(frame, [hull], -1, COLOR_EYE, 1)

        # Mouth contour
        mouth_pts = coords[MOUTH_ALL_IDX]
        hull = cv2.convexHull(mouth_pts)
        cv2.drawContours(frame, [hull], -1, COLOR_MOUTH, 1)

        # Minimal mesh (optional)
        if draw_mesh:
            for pt in coords[::6]:   # every 6th point to keep it light
                cv2.circle(frame, tuple(pt), 1, (80, 80, 80), -1)

    def get_face_bbox(self, coords):
        """Return (x, y, w, h) bounding box of entire face."""
        x, y, w, h = cv2.boundingRect(coords)
        return x, y, w, h

"""
app.py — Streamlit Web App for Driver Drowsiness Detection

The webcam is opened directly on the local machine with OpenCV (no WebRTC,
no browser camera permissions, no STUN/TURN negotiation). Capture and
detection run continuously in a background thread — decoupled from
Streamlit's script-rerun cycle — and are served to the browser as a plain
MJPEG stream over a small local HTTP server. The browser renders that
stream natively (exactly like a live IP camera feed), so the video is not
subject to Streamlit's per-rerun clear-and-redraw behavior, which is what
previously caused visible flicker and capped the achievable frame rate.
Only the small text/metric widgets are updated via a (much cheaper)
Streamlit fragment.

Usage:
    streamlit run app.py
"""

import platform
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf

from utils.face_utils import FaceAnalyzer, RIGHT_EAR_PTS, LEFT_EAR_PTS

# ── State constants ────────────────────────────────────────────────────────────
STATE_AWAKE  = 0
STATE_CLOSED = 1
STATE_YAWN   = 2

STATE_NAMES = {
    STATE_AWAKE:  'AWAKE',
    STATE_CLOSED: 'EYES CLOSED',
    STATE_YAWN:   'YAWNING',
}

STATE_COLORS_BGR = {
    STATE_AWAKE:  (0, 220, 0),
    STATE_CLOSED: (0, 0, 255),
    STATE_YAWN:   (255, 165, 0),
}

DROWSY_STATES = {STATE_CLOSED}

# ── Detection thresholds ───────────────────────────────────────────────────────
EAR_THRESHOLD   = 0.20
MAR_THRESHOLD   = 0.55
CONF_THRESHOLD  = 0.60
ALARM_SECS      = 2.5
MODEL_PATH      = 'models/drowsiness_model_ft.keras'
LANDMARKER_PATH = 'face_landmarker.task'

# ── Camera / performance tuning ────────────────────────────────────────────────
FRAME_W, FRAME_H     = 640, 480
METRICS_REFRESH_SECS = 0.3     # cadence for the (lightweight, text-only) metrics fragment
SLOW_REFRESH_SECS    = 1.0     # cadence for the session-statistics panel
CNN_EVERY_N_FRAMES   = 3       # only run the CNN on every Nth ambiguous frame
MAX_CONSEC_FAILURES  = 20      # ~ a few seconds of dropped frames before giving up
MJPEG_QUALITY        = 80
IS_WINDOWS = platform.system() == 'Windows'

# Guards MediaPipe / TensorFlow calls. Only the single background CameraWorker
# thread ever calls these in normal operation; the lock is cheap insurance in
# case a second worker is ever spun up.
_inference_lock = threading.Lock()


# ── Cached resource loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_tf_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    size  = model.input_shape[1]
    model.predict(np.zeros((1, size, size, 3), dtype=np.float32), verbose=0)
    return model


@st.cache_resource(show_spinner=False)
def load_face_analyzer():
    return FaceAnalyzer(model_path=LANDMARKER_PATH)


def load_models_safely():
    """Load (and warm up) the model + landmarker once per process.

    Returns (model, model_size, analyzer, error_message). On failure the
    first three values are None and error_message describes what broke.
    """
    try:
        model = load_tf_model()
        analyzer = load_face_analyzer()
        return model, model.input_shape[1], analyzer, None
    except Exception as exc:
        return None, None, None, f"{type(exc).__name__}: {exc}"


# ── Session-level drowsiness tracker ───────────────────────────────────────────
DROWSY_GRACE_SECS = 0.5   # tolerate brief interruptions without resetting the alarm timer


class SessionTracker:
    def __init__(self, threshold=ALARM_SECS, window=8):
        import collections
        self.threshold      = threshold
        self.buffer         = collections.deque(maxlen=window)
        self.drowsy_start   = None
        self.last_drowsy_at = None
        self.alarm_on       = False
        self.alarm_count    = 0
        self.frames         = 0
        self.durations      = collections.defaultdict(float)
        self._tick          = time.time()

    def update(self, raw):
        import collections
        now = time.time()
        dt  = now - self._tick
        self._tick = now
        self.frames += 1

        self.buffer.append(raw)
        smoothed = int(collections.Counter(self.buffer).most_common(1)[0][0])
        self.durations[smoothed] += dt

        secs = 0.0
        if smoothed in DROWSY_STATES:
            self.last_drowsy_at = now
            if self.drowsy_start is None:
                self.drowsy_start = now
            secs = now - self.drowsy_start
            if secs >= self.threshold and not self.alarm_on:
                self.alarm_on = True
                self.alarm_count += 1
        elif self.drowsy_start is not None:
            # A single missed detection or borderline EAR reading can flip the
            # smoothed majority away from CLOSED for a frame or two even while
            # the eyes are still shut -- don't let that fully restart the 2.5s
            # countdown. Only reset once eyes have genuinely looked open/away
            # for longer than the grace period.
            if now - self.last_drowsy_at > DROWSY_GRACE_SECS:
                self.drowsy_start = None
                self.alarm_on = False
            else:
                secs = now - self.drowsy_start

        return smoothed, self.alarm_on, secs

    def get_stats(self):
        total = sum(self.durations.values()) or 1
        return {
            'frames': self.frames,
            'alarms': self.alarm_count,
            'percentages': {
                STATE_NAMES.get(k, '?'): round(v / total * 100, 1)
                for k, v in self.durations.items()
                if k in STATE_NAMES
            },
        }


# ── Camera handling ─────────────────────────────────────────────────────────────
MAX_ACCEPTABLE_READ_SECS = 0.3   # a healthy webcam read takes a few ms, not this


def open_camera(index: int):
    """Open the local webcam. Returns (cap, error_message).

    Backend choice matters a lot more than backend *name*: on some Windows
    machines DirectShow (CAP_DSHOW) opens instantly but then blocks for a
    full second on every single read() (and returns near-black frames) on
    a given camera/driver combo, while the platform default is fast; on
    others it's the reverse. Rather than hard-coding one backend, each
    candidate is opened and a real frame is timed — whichever backend
    actually delivers frames quickly is the one that gets used.
    """
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW] if IS_WINDOWS else [cv2.CAP_ANY]
    last_err = f"Could not open camera index {index}."
    fallback = None   # (cap, elapsed) — kept only if no fast backend is found

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        t0 = time.time()
        ok, frame = cap.read()
        elapsed = time.time() - t0

        if not ok or frame is None:
            cap.release()
            last_err = "Camera opened but did not return any frames."
            continue

        if elapsed <= MAX_ACCEPTABLE_READ_SECS:
            if fallback is not None:
                fallback[0].release()
            return cap, None

        if fallback is None or elapsed < fallback[1]:
            if fallback is not None:
                fallback[0].release()
            fallback = (cap, elapsed)
        else:
            cap.release()

    if fallback is not None:
        cap, elapsed = fallback
        if elapsed <= MAX_ACCEPTABLE_READ_SECS * 2:
            return cap, None
        cap.release()
        return None, (
            "Camera responded too slowly to stream live video "
            "(it may be in use by another app, browser tab, or a previous "
            "unclosed run of this app). Close other apps using the camera "
            "and press Start again."
        )
    return None, last_err


# ── Per-frame detection ─────────────────────────────────────────────────────────
def process_frame(frame, model, model_size, analyzer, tracker, cnn_state):
    """Run detection on one BGR frame. Mutates `frame` in place with the HUD.

    `cnn_state` is a small dict (`{'counter': int, 'last_raw': int, 'last_conf': float}`)
    persisted across calls so the CNN only runs every CNN_EVERY_N_FRAMES on
    ambiguous (not-clearly-closed, not-clearly-yawning) frames.
    """
    frame = cv2.flip(frame, 1)

    with _inference_lock:
        result = analyzer.detect(frame)
    coords = analyzer.get_pixel_coords(result, frame.shape)
    found  = coords is not None

    raw, conf, ear, mar = STATE_AWAKE, 1.0, 0.0, 0.0

    if found:
        analyzer.draw_landmarks(frame, coords)
        ear = (analyzer.eye_aspect_ratio(coords, RIGHT_EAR_PTS) +
               analyzer.eye_aspect_ratio(coords, LEFT_EAR_PTS)) / 2.0
        mar = analyzer.mouth_aspect_ratio(coords)

        if ear < EAR_THRESHOLD:
            raw, conf = STATE_CLOSED, 1.0
        elif mar >= MAR_THRESHOLD:
            raw, conf = STATE_YAWN, 1.0
        else:
            cnn_state['counter'] += 1
            if cnn_state['counter'] % CNN_EVERY_N_FRAMES == 0:
                fx, fy, fw, fh = analyzer.get_face_bbox(coords)
                hf, wf = frame.shape[:2]
                pad = 20
                roi = frame[max(0, fy - pad):min(hf, fy + fh + pad),
                            max(0, fx - pad):min(wf, fx + fw + pad)]
                if roi.size > 0:
                    r = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    r = cv2.resize(r, (model_size, model_size)).astype(np.float32) / 255.0
                    with _inference_lock:
                        pred = model.predict(np.expand_dims(r, 0), verbose=0)[0]
                    yawn_conf, awake_conf = float(pred[0]), float(pred[1])
                    if yawn_conf >= CONF_THRESHOLD:
                        raw, conf = STATE_YAWN, yawn_conf
                    else:
                        raw, conf = STATE_AWAKE, awake_conf
                    cnn_state['last_raw']  = raw
                    cnn_state['last_conf'] = conf
            else:
                raw  = cnn_state.get('last_raw', STATE_AWAKE)
                conf = cnn_state.get('last_conf', 1.0)

    smoothed, alarm, secs = tracker.update(raw)
    draw_hud(frame, smoothed, alarm, secs, conf, found, ear, mar)

    metrics = {
        'state': smoothed, 'ear': ear, 'mar': mar, 'conf': conf,
        'alarm': alarm, 'drowsy_secs': secs, 'found': found,
    }
    return frame, metrics


def draw_hud(f, state, alarm, secs, conf, found, ear, mar):
    h, w = f.shape[:2]
    color = STATE_COLORS_BGR.get(state, (200, 200, 200))
    label = STATE_NAMES.get(state, 'UNKNOWN')

    ov = f.copy()
    cv2.rectangle(ov, (10, 10), (370, 125), (10, 10, 30), -1)
    cv2.addWeighted(ov, 0.6, f, 0.4, 0, f)
    cv2.rectangle(f, (10, 10), (370, 125), color, 1)

    cv2.putText(f, label, (22, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.15, color, 2, cv2.LINE_AA)
    cv2.putText(f, f"Conf: {conf*100:.0f}%", (22, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 220), 1)
    cv2.putText(f, f"EAR: {ear:.3f}   MAR: {mar:.3f}", (22, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 180), 1)

    if secs > 0:
        bar_w = w - 44
        fill  = min(secs / ALARM_SECS, 1.0)
        bar_c = (0, 60, 255) if fill >= 1.0 else (0, 140, 255)
        cv2.rectangle(f, (22, h - 32), (22 + bar_w, h - 18), (40, 40, 60), -1)
        cv2.rectangle(f, (22, h - 32), (22 + int(bar_w * fill), h - 18), bar_c, -1)
        cv2.putText(f, f"Drowsy: {secs:.1f}s / {ALARM_SECS}s", (22, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, bar_c, 1)

    if alarm and int(time.time() * 2) % 2 == 0:
        cv2.rectangle(f, (0, 0), (w, h), (0, 30, 220), 8)
        cv2.rectangle(f, (w // 2 - 130, h // 2 - 40), (w // 2 + 130, h // 2 + 20), (0, 0, 0), -1)
        cv2.putText(f, 'WAKE UP!', (w // 2 - 118, h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 60, 255), 4, cv2.LINE_AA)

    if not found:
        cv2.putText(f, 'No face detected', (w // 2 - 135, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 255), 2)


# ── Background capture/detection worker ─────────────────────────────────────────
class CameraWorker:
    """Owns the camera and runs capture + detection in its own thread.

    This loop is driven purely by how fast the camera and models can go —
    it is NOT paced by Streamlit's rerun cycle — so the achievable frame
    rate is limited only by real hardware/processing cost, not by any
    artificial polling interval. Each finished frame is JPEG-encoded and
    exposed to the local MJPEG server, plus a metrics snapshot exposed to
    the (much lighter) Streamlit metrics fragment.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._frame_id = 0
        self._metrics = None
        self._error = None
        self.running = False
        self._stop_evt = threading.Event()
        self._thread = None
        self._cap = None
        self.tracker = None
        self._cnn_state = None
        self._model = None
        self._model_size = None
        self._analyzer = None

    def start(self, camera_index):
        if self.running:
            return None

        # Load (or fetch the already-cached) model/analyzer here, on the
        # caller's thread, which is Streamlit's own script-run thread. These
        # loaders are decorated with @st.cache_resource, and Streamlit's
        # caching layer assumes it is being called from a thread with an
        # active script-run context — calling it from the background worker
        # thread instead can hang indefinitely. The background thread only
        # ever touches plain TensorFlow/MediaPipe/OpenCV calls, never `st.*`.
        model, model_size, analyzer, err = load_models_safely()
        if err:
            return f"Failed to load model: {err}"

        cap, err = open_camera(camera_index)
        if err:
            return err

        self._model = model
        self._model_size = model_size
        self._analyzer = analyzer
        self._cap = cap
        self._stop_evt.clear()
        self._error = None
        self._jpeg = None
        self._metrics = None
        self.tracker = SessionTracker()
        self._cnn_state = {'counter': 0, 'last_raw': STATE_AWAKE, 'last_conf': 1.0}
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return None

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._release_camera()
        self.running = False
        with self._cond:
            self._jpeg = None
            self._cond.notify_all()

    def pop_error(self):
        err = self._error
        self._error = None
        return err

    def _release_camera(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _loop(self):
        model, model_size, analyzer = self._model, self._model_size, self._analyzer
        fps_ema = 0.0
        last_t = time.time()
        failures = 0

        while not self._stop_evt.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= MAX_CONSEC_FAILURES:
                    self._error = "Camera feed lost. Check the connection and press Start again."
                    break
                continue
            failures = 0

            now = time.time()
            dt = now - last_t
            last_t = now
            fps_ema = 0.9 * fps_ema + 0.1 / max(dt, 1e-6)

            annotated, metrics = process_frame(
                frame, model, model_size, analyzer, self.tracker, self._cnn_state,
            )
            metrics['fps'] = fps_ema
            metrics['stats'] = self.tracker.get_stats()

            ok2, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
            jpeg = buf.tobytes() if ok2 else None

            with self._cond:
                self._jpeg = jpeg
                self._frame_id += 1
                self._metrics = metrics
                self._cond.notify_all()

        self._release_camera()
        self.running = False

    def get_latest_jpeg(self, last_id, timeout=1.0):
        """Block until a frame newer than `last_id` is available (or timeout)."""
        with self._cond:
            if self._frame_id == last_id:
                self._cond.wait(timeout=timeout)
            return self._jpeg, self._frame_id

    def get_metrics(self):
        return self._metrics


@st.cache_resource(show_spinner=False)
def get_camera_worker():
    return CameraWorker()


def _make_stream_handler(worker):
    class StreamHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split('?')[0] != '/stream':
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            last_id = -1
            try:
                while True:
                    jpeg, last_id = worker.get_latest_jpeg(last_id, timeout=1.0)
                    if jpeg is None:
                        if not worker.running:
                            break
                        continue
                    self.wfile.write(
                        b'--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpeg)
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, fmt, *args):
            pass   # silence default per-request logging to stdout

    return StreamHandler


@st.cache_resource(show_spinner=False)
def get_stream_server():
    """Local loopback-only HTTP server that serves the MJPEG stream. Created
    once per process and kept alive for the app's lifetime."""
    worker = get_camera_worker()
    handler_cls = _make_stream_handler(worker)
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_cls)
    server.daemon_threads = True
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def is_running():
    return get_camera_worker().running


# ── Session state / lifecycle ───────────────────────────────────────────────────
def init_state():
    defaults = {'camera_index': 0, 'error': None, 'stream_token': 0}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def start_detection():
    worker = get_camera_worker()
    if worker.running:
        return   # guards against a double-click opening a second camera handle
    get_stream_server()   # make sure the MJPEG server is up before we point <img> at it
    err = worker.start(st.session_state.camera_index)
    if err:
        st.session_state.error = err
        return
    st.session_state.error = None
    st.session_state.stream_token += 1


def stop_detection(error=None):
    get_camera_worker().stop()
    if error:
        st.session_state.error = error


# ── Browser alarm sound via Web Audio API ─────────────────────────────────────
def _render_alarm_sound(alarm_active: bool):
    js = f"""
    <script>
    (function() {{
        const ACTIVE = {'true' if alarm_active else 'false'};
        const KEY    = '__ddd_alarm_interval__';
        const CTXKEY = '__ddd_alarm_ctx__';

        if (!ACTIVE) {{
            if (window[KEY]) {{ clearInterval(window[KEY]); window[KEY] = null; }}
            return;
        }}
        if (window[KEY]) return;

        const AC = window.AudioContext || window.webkitAudioContext;
        if (!AC) return;
        if (!window[CTXKEY]) window[CTXKEY] = new AC();
        const ctx = window[CTXKEY];

        function beep(freq, start, dur) {{
            const osc  = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(freq, ctx.currentTime + start);
            gain.gain.setValueAtTime(0.35, ctx.currentTime + start);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + dur);
            osc.start(ctx.currentTime + start);
            osc.stop(ctx.currentTime + start + dur + 0.05);
        }}

        function playPattern() {{
            beep(960, 0.00, 0.18);
            beep(720, 0.22, 0.18);
            beep(960, 0.44, 0.18);
            beep(720, 0.66, 0.18);
        }}

        playPattern();
        window[KEY] = setInterval(playPattern, 1000);
    }})();
    </script>
    """
    components.html(js, height=0)


# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', 'Segoe UI', sans-serif !important; }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

.stApp {
    background: radial-gradient(ellipse at 15% -10%, #0c2340 0%, #060f22 45%, #04060f 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071021 0%, #04080f 100%) !important;
    border-right: 1px solid rgba(34,211,238,0.12);
}
[data-testid="stSidebar"] * { color: #c3d0ea !important; }
[data-testid="stSidebarContent"] { padding: 1.2rem 1rem; }

/* ── Header ── */
h1 {
    background: linear-gradient(135deg, #22d3ee 0%, #3b82f6 55%, #818cf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    font-weight: 800 !important; font-size: 2.15rem !important; letter-spacing: -0.5px; line-height: 1.2 !important;
    margin-bottom: 0 !important;
}
h2, h3, h4, h5 { color: #e4ecfb !important; font-weight: 600 !important; }
p, li, span, label { color: #93a4c9 !important; }

/* ── Status pill (running / idle) ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 16px; border-radius: 999px; font-weight: 700; font-size: 0.78rem;
    letter-spacing: 0.6px; text-transform: uppercase; float: right; margin-top: 0.6rem;
}
.status-running { background: rgba(34,211,238,0.12); color: #22d3ee !important; border: 1px solid rgba(34,211,238,0.4); }
.status-idle     { background: rgba(148,163,184,0.10); color: #94a3b8 !important; border: 1px solid rgba(148,163,184,0.3); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-running { background: #22d3ee; box-shadow: 0 0 8px #22d3ee; animation: pulse-cyan 1.1s infinite; }
.dot-idle    { background: #94a3b8; }
@keyframes pulse-cyan { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

/* ── Glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(148,197,255,0.12);
    border-radius: 18px; padding: 1.2rem 1.4rem;
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
}

/* ── Status badges ── */
.badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 16px; border-radius: 999px; font-weight: 600; font-size: 0.88rem; letter-spacing: 0.4px;
}
.badge-awake  { background: rgba(34,211,140,0.15);  color: #34d399 !important; border: 1px solid rgba(34,211,140,0.3); }
.badge-closed { background: rgba(255,60,60,0.15);   color: #f87171 !important; border: 1px solid rgba(255,60,60,0.3); }
.badge-yawn   { background: rgba(251,146,60,0.15);  color: #fb923c !important; border: 1px solid rgba(251,146,60,0.3); }
.badge-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.dot-awake  { background: #34d399; box-shadow: 0 0 6px #34d399; }
.dot-closed { background: #f87171; box-shadow: 0 0 6px #f87171; animation: pulse-red 0.8s infinite; }
.dot-yawn   { background: #fb923c; box-shadow: 0 0 6px #fb923c; }
@keyframes pulse-red { 0%,100% { box-shadow: 0 0 4px #f87171; } 50% { box-shadow: 0 0 12px #f87171, 0 0 24px rgba(248,113,113,0.4); } }

/* ── Alarm banner ── */
.alarm-banner {
    background: linear-gradient(135deg, rgba(220,38,38,0.25), rgba(185,28,28,0.15));
    border: 2px solid rgba(239,68,68,0.6); border-radius: 14px; padding: 1rem 1.5rem; text-align: center;
    animation: alarm-flash 0.6s infinite alternate; color: #fca5a5 !important;
    font-weight: 700 !important; font-size: 1.1rem !important; letter-spacing: 1px;
}
@keyframes alarm-flash {
    from { border-color: rgba(239,68,68,0.4); background: rgba(220,38,38,0.15); }
    to   { border-color: rgba(239,68,68,0.9); background: rgba(220,38,38,0.35); box-shadow: 0 0 20px rgba(239,68,68,0.3); }
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.045) !important;
    border: 1px solid rgba(148,197,255,0.12) !important;
    border-radius: 16px !important; padding: 1rem 1.2rem !important;
    backdrop-filter: blur(8px); transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
[data-testid="stMetricLabel"] { color: #7d8fbd !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.6px; }
[data-testid="stMetricValue"] { color: #e8eefc !important; font-weight: 700 !important; font-size: 1.55rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ── Table ── */
table { width: 100%; border-collapse: collapse; }
th, td { padding: 7px 12px; font-size: 0.85rem; }
th { color: #5f9fd6 !important; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.7px; border-bottom: 1px solid rgba(255,255,255,0.08); }
td { color: #c8d4f0 !important; border-bottom: 1px solid rgba(255,255,255,0.05); }
tr:last-child td { border-bottom: none; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 12px !important; font-weight: 600 !important;
    transition: opacity 0.2s, transform 0.2s, box-shadow 0.2s !important;
    border: none !important;
}
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #06b6d4, #3b82f6) !important; color: #fff !important;
    box-shadow: 0 4px 18px rgba(6,182,212,0.35) !important;
}
[data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,0.06) !important; color: #cbd5f0 !important;
    border: 1px solid rgba(148,163,184,0.3) !important;
}
.stButton > button:hover:not(:disabled) { opacity: 0.9 !important; transform: translateY(-1px) !important; }
.stButton > button:disabled { opacity: 0.4 !important; }

/* ── Misc ── */
.stAlert { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(100,180,255,0.18) !important; border-radius: 12px !important; color: #9db3d8 !important; }
hr { border-color: rgba(255,255,255,0.07) !important; }
.stCaption, .stMarkdown small { color: #56618a !important; }

/* ── Camera preview ── */
[data-testid="stImage"] img, .video-stream {
    border-radius: 18px !important;
    border: 1px solid rgba(34,211,238,0.25) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
}
.video-stream { width: 100%; height: auto; display: block; background: #05070d; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg,#06b6d4,#3b82f6) !important; border-radius: 4px !important; }

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase;
    color: #4d7cad !important; margin-bottom: 0.5rem; display: block;
}

/* ── Number input in sidebar ── */
[data-testid="stNumberInput"] input { border-radius: 8px !important; }

/* ── Footer ── */
.footer { text-align: center; color: #2d3a5a !important; font-size: 0.75rem; padding: 2rem 0 1rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 2rem; }
.footer span { color: #22d3ee !important; }
</style>
"""


# ── Sidebar ─────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        st.markdown("---")

        running = is_running()

        st.markdown('<span class="section-label">Camera Source</span>', unsafe_allow_html=True)
        st.session_state.camera_index = st.number_input(
            "Camera index", min_value=0, max_value=10,
            value=st.session_state.camera_index, step=1,
            disabled=running, label_visibility="collapsed",
        )

        col_start, col_stop = st.columns(2)
        start_clicked = col_start.button(
            "▶ Start", type="primary", width="stretch",
            disabled=running,
        )
        stop_clicked = col_stop.button(
            "⏹ Stop", width="stretch",
            disabled=not running,
        )

        if start_clicked:
            with st.spinner("Starting camera & loading model…"):
                start_detection()
            st.rerun()
        if stop_clicked:
            stop_detection()
            st.rerun()

        if st.session_state.error:
            st.error(f"⚠️ {st.session_state.error}")

        st.markdown("---")
        st.markdown('<span class="section-label">Detection Thresholds</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card">
        <table>
        <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody>
        <tr><td>EAR (eye closed)</td><td><b>&lt; {EAR_THRESHOLD}</b></td></tr>
        <tr><td>MAR (yawning)</td><td><b>&gt; {MAR_THRESHOLD}</b></td></tr>
        <tr><td>CNN confidence</td><td><b>&gt; {CONF_THRESHOLD*100:.0f}%</b></td></tr>
        <tr><td>Alarm delay</td><td><b>{ALARM_SECS}s</b></td></tr>
        </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<span class="section-label">How It Works</span>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card" style="font-size:0.85rem; line-height:1.8;">
        <b style="color:#e2e8f8;">1.</b> MediaPipe detects 468 face landmarks<br>
        <b style="color:#e2e8f8;">2.</b> EAR computed from 6 eye points<br>
        <b style="color:#e2e8f8;">3.</b> MAR computed from 8 mouth points<br>
        <b style="color:#e2e8f8;">4.</b> MobileNetV2 CNN validates ambiguous frames<br>
        <b style="color:#e2e8f8;">5.</b> 8-frame temporal smoothing filters noise<br>
        <b style="color:#e2e8f8;">6.</b> Eyes closed &gt; 2.5s &rarr; 🔴 Audio alarm
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<span class="section-label">State Legend</span>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
        <div style="margin-bottom:10px;">
            <span class="badge badge-awake"><span class="badge-dot dot-awake"></span>AWAKE</span>
            <span style="font-size:0.8rem; color:#4a5a8a; margin-left:8px;">Alert &amp; attentive</span>
        </div>
        <div style="margin-bottom:10px;">
            <span class="badge badge-closed"><span class="badge-dot dot-closed"></span>EYES CLOSED</span>
            <span style="font-size:0.8rem; color:#4a5a8a; margin-left:8px;">Triggers alarm</span>
        </div>
        <div>
            <span class="badge badge-yawn"><span class="badge-dot dot-yawn"></span>YAWNING</span>
            <span style="font-size:0.8rem; color:#4a5a8a; margin-left:8px;">Logged, no alarm</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="footer" style="margin-top:1rem;">
        <span>TensorFlow</span> · <span>MediaPipe</span> · <span>OpenCV</span>
        </div>
        """, unsafe_allow_html=True)


# ── Live video (native MJPEG stream, no Streamlit rerun involved) ──────────────
def render_live():
    running = is_running()
    col_video, col_panel = st.columns([2.2, 1], gap="large")

    with col_video:
        st.markdown('<span class="section-label">🎥 Live Detection Feed</span>', unsafe_allow_html=True)
        if running:
            _, port = get_stream_server()
            st.markdown(
                f'<img class="video-stream" '
                f'src="http://127.0.0.1:{port}/stream?t={st.session_state.stream_token}">',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:4rem 2rem; color:#4a5a8a;">
                <div style="font-size:3rem; margin-bottom:0.8rem;">📷</div>
                <div style="font-weight:600; color:#8899bb; margin-bottom:0.4rem; font-size:1.05rem;">Camera Inactive</div>
                <div style="font-size:0.85rem;">Click <b style="color:#22d3ee;">▶ Start</b> in the sidebar to begin detection</div>
            </div>
            """, unsafe_allow_html=True)

    with col_panel:
        st.markdown('<span class="section-label">📡 Live Metrics</span>', unsafe_allow_html=True)
        render_metrics_fragment()


# ── Live metrics fragment (lightweight text/widgets only — no image payload) ───
@st.fragment(run_every=METRICS_REFRESH_SECS)
def render_metrics_fragment():
    worker = get_camera_worker()

    if not worker.running:
        _render_alarm_sound(False)
        err = worker.pop_error()
        if err:
            stop_detection(error=err)
            st.rerun()
        st.markdown("""
        <div class="glass-card" style="text-align:center; color:#4a5a8a; padding:2rem;">
            Metrics will appear here once detection starts.
        </div>
        """, unsafe_allow_html=True)
        return

    metrics = worker.get_metrics()
    if metrics is None:
        st.markdown("""
        <div class="glass-card" style="text-align:center; color:#4a5a8a; padding:2rem;">
            Waiting for the first frame…
        </div>
        """, unsafe_allow_html=True)
        return

    # Checked on the fast cadence (not the slower stats fragment) so the audio
    # cue starts within ~METRICS_REFRESH_SECS of the alarm actually firing,
    # instead of waiting for the next once-a-second stats refresh.
    _render_alarm_sound(bool(metrics.get('alarm')))
    render_metrics_panel(metrics)


def render_metrics_panel(metrics):
    s = metrics['state']
    badge_cls = {STATE_AWAKE: 'badge-awake', STATE_CLOSED: 'badge-closed', STATE_YAWN: 'badge-yawn'}.get(s, 'badge-awake')
    dot_cls   = {STATE_AWAKE: 'dot-awake', STATE_CLOSED: 'dot-closed', STATE_YAWN: 'dot-yawn'}.get(s, 'dot-awake')
    state_name = STATE_NAMES.get(s, 'UNKNOWN')

    st.markdown(f"""
    <div class="glass-card" style="text-align:center; padding:1.4rem;">
        <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;color:#4a5a8a;margin-bottom:0.6rem;">Current State</div>
        <span class="badge {badge_cls}" style="font-size:1rem; padding:8px 20px;">
            <span class="badge-dot {dot_cls}"></span>{state_name}
        </span>
    </div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    m1.metric("👁️ EAR", f"{metrics['ear']:.3f}",
              delta="closed" if metrics['ear'] < EAR_THRESHOLD else "open",
              delta_color="inverse" if metrics['ear'] < EAR_THRESHOLD else "normal")
    m2.metric("👄 MAR", f"{metrics['mar']:.3f}",
              delta="yawn" if metrics['mar'] >= MAR_THRESHOLD else "normal",
              delta_color="inverse" if metrics['mar'] >= MAR_THRESHOLD else "off")

    m3, m4 = st.columns(2)
    m3.metric("🎯 Conf", f"{metrics['conf']*100:.0f}%")
    m4.metric("⚡ FPS", f"{metrics['fps']:.1f}")

    if metrics['drowsy_secs'] > 0:
        st.markdown('<span class="section-label" style="margin-top:0.6rem;">⏱ Drowsy Duration</span>', unsafe_allow_html=True)
        st.progress(min(metrics['drowsy_secs'] / ALARM_SECS, 1.0))
        st.caption(f"{metrics['drowsy_secs']:.1f}s of {ALARM_SECS}s threshold")

    if metrics['alarm']:
        st.markdown("""
        <div class="alarm-banner">
            ⚠️ &nbsp; DROWSINESS ALARM &nbsp; ⚠️<br>
            <small style="font-weight:400;font-size:0.8rem;color:#fca5a5;">Please pull over safely</small>
        </div>
        """, unsafe_allow_html=True)


# ── Session stats fragment (slow — the alarm sound is handled by the fast
#    metrics fragment above so it isn't gated behind this slower cadence) ───────
@st.fragment(run_every=SLOW_REFRESH_SECS)
def render_session_stats():
    worker = get_camera_worker()
    metrics = worker.get_metrics() if worker.running else None

    st.markdown('<span class="section-label">📊 Session Statistics</span>', unsafe_allow_html=True)

    if worker.running and metrics and 'stats' in metrics:
        stats = metrics['stats']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🖼️ Total Frames", f"{stats['frames']:,}")
        c2.metric("🔔 Alarm Events", stats['alarms'],
                  delta="high" if stats['alarms'] > 3 else None,
                  delta_color="inverse" if stats['alarms'] > 3 else "off")

        pcts = stats['percentages']
        awake_pct  = pcts.get('AWAKE', 0)
        closed_pct = pcts.get('EYES CLOSED', 0)
        yawn_pct   = pcts.get('YAWNING', 0)

        c3.metric("😊 Awake", f"{awake_pct}%")
        c4.metric("😴 Eyes Closed", f"{closed_pct}%",
                  delta="⚠ risky" if closed_pct > 20 else None,
                  delta_color="inverse" if closed_pct > 20 else "off")

        if yawn_pct > 0:
            st.progress(yawn_pct / 100, text=f"🥱 Yawning: {yawn_pct}% of session")
        if awake_pct > 0:
            st.progress(awake_pct / 100, text=f"😊 Awake: {awake_pct}% of session")
        if closed_pct > 0:
            st.progress(closed_pct / 100, text=f"😴 Eyes Closed: {closed_pct}% of session")
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; color:#4a5a8a; padding:1.8rem;">
            Statistics will appear here once detection starts.
        </div>
        """, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="DrowseGuard — Driver Safety AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_state()
    st.markdown(_CSS, unsafe_allow_html=True)
    render_sidebar()

    running = is_running()

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.title("🛡️ DrowseGuard")
        st.caption("Real-time Driver Drowsiness Detection  ·  TensorFlow + MediaPipe  ·  EAR / MAR + MobileNetV2 CNN")
    with col_badge:
        pill_cls = "status-running" if running else "status-idle"
        dot_cls  = "dot-running" if running else "dot-idle"
        pill_txt = "Running" if running else "Idle"
        st.markdown(f"""
        <div style="text-align:right; padding-top:1.2rem;">
            <span class="status-pill {pill_cls}"><span class="status-dot {dot_cls}"></span>{pill_txt}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Live feed / metrics ──────────────────────────────────────────────────
    render_live()

    st.markdown("---")

    # ── Alarm sound + session statistics ────────────────────────────────────
    render_session_stats()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        Built with <span>Streamlit</span> · <span>TensorFlow</span> · <span>MediaPipe</span> · <span>OpenCV</span><br>
        Desktop app: <code style="color:#22d3ee;">python detect.py</code>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

"""
utils/state_tracker.py
Drowsiness state tracker with temporal smoothing and 3-level alarm triggering.

Alarm levels (eyes-closed only):
  NORMAL  — eyes closed ≥ 2 s for the first time (or isolated event)
  SHORT   — repeated closures (≥ REPEAT_MIN events in REPEAT_WINDOW seconds)
  POWER   — single continuous closure ≥ POWER_SECS (severely drowsy)

Audio files (place in audio/ next to detect.py):
  audio/normal_alarm.wav
  audio/short_alarm.mp3
  audio/power_alarm.wav
"""

import time
import collections
import os
import numpy as np
import pygame


# ─── State definitions (must match CLASS_NAMES order in data_utils.py) ─────────
# CLASS_NAMES = ['awake', 'yawning']   — CLOSED is rule-based, not CNN
STATE_AWAKE  = 0
STATE_CLOSED = 1
STATE_YAWN   = 2

STATE_NAMES = {
    STATE_AWAKE:  'AWAKE',
    STATE_CLOSED: 'EYES CLOSED',
    STATE_YAWN:   'YAWNING',
}

STATE_COLORS_BGR = {
    STATE_AWAKE:  (0, 220, 0),    # Green
    STATE_CLOSED: (0, 0, 255),    # Red
    STATE_YAWN:   (255, 165, 0),  # Orange
}

# Eyes-closed triggers the alarm (yawning does not)
DROWSY_STATES = {STATE_CLOSED}

# ─── Alarm thresholds ────────────────────────────────────────────────────────
ALARM_SECS    = 2.0    # eyes closed this long → first alarm fires
POWER_SECS    = 5.0    # eyes closed this long continuously → power alarm
REPEAT_WINDOW = 60.0   # seconds to look back when counting past alarms
REPEAT_MIN    = 2      # ≥ this many alarms in REPEAT_WINDOW → short-alarm mode

# Smoothing window (frames)
SMOOTH_WINDOW = 8

# Alarm level constants
LEVEL_NONE   = 0
LEVEL_NORMAL = 1   # isolated 2-sec closure
LEVEL_SHORT  = 2   # repeated microsleep pattern
LEVEL_POWER  = 3   # sustained closure > POWER_SECS

# Audio file paths (relative to working directory, i.e. project root)
_AUDIO = {
    LEVEL_NORMAL: os.path.join('audio', 'normal_alarm.wav'),
    LEVEL_SHORT:  os.path.join('audio', 'short_alarm.mp3'),
    LEVEL_POWER:  os.path.join('audio', 'power_alarm.wav'),
}

# Fallback beep params if file is missing: (freq_hz, duration_s, volume)
_BEEP_FALLBACK = {
    LEVEL_NORMAL: (880,  0.6, 0.8),
    LEVEL_SHORT:  (1200, 0.2, 1.0),
    LEVEL_POWER:  (660,  1.2, 1.0),
}


class DrowsinessTracker:
    def __init__(self,
                 alarm_sound_path=None,   # kept for API compat, unused
                 alarm_threshold=ALARM_SECS,
                 smooth_window=SMOOTH_WINDOW):
        self.alarm_threshold = alarm_threshold
        self.smooth_window   = smooth_window

        self.pred_buffer = collections.deque(maxlen=smooth_window)

        # Timing
        self.drowsy_start_time  = None
        self.current_level      = LEVEL_NONE   # active alarm level this closure
        self.alarm_history      = []           # timestamps of past alarm fires
        self.alarm_count        = 0            # total alarms this session

        # Stats
        self.frame_count     = 0
        self.state_durations = collections.defaultdict(float)
        self._last_tick      = time.time()

        self._init_audio()

    # ── Audio setup ──────────────────────────────────────────────────────────
    def _init_audio(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._audio_ok = True
        except Exception as e:
            print(f"[Audio] Could not initialise pygame.mixer: {e}")
            self._audio_ok = False
            return

        self._sounds = {}
        for level in (LEVEL_NORMAL, LEVEL_SHORT, LEVEL_POWER):
            self._sounds[level] = self._load_sound(level)

    def _load_sound(self, level):
        path = _AUDIO[level]
        if os.path.exists(path):
            try:
                snd = pygame.mixer.Sound(path)
                snd.set_volume(_BEEP_FALLBACK[level][2])
                print(f"[Audio] Loaded {path}")
                return snd
            except Exception:
                pass  # fall through to generated beep
        # Generate beep fallback
        freq, dur, vol = _BEEP_FALLBACK[level]
        print(f"[Audio] {path} not found — using {freq}Hz generated beep")
        return self._generate_beep(freq, dur, vol)

    def _generate_beep(self, freq=880, duration=0.5, volume=0.8, sample_rate=44100):
        t    = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        snd  = pygame.sndarray.make_sound(wave)
        snd.set_volume(volume)
        return snd

    # ── Per-frame update ─────────────────────────────────────────────────────
    def update(self, raw_prediction: int):
        """
        Feed a raw class index each frame.
        Returns (smoothed_state, alarm_triggered, seconds_closed)
        """
        now = time.time()
        dt  = now - self._last_tick
        self._last_tick = now
        self.frame_count += 1

        # Temporal smoothing
        self.pred_buffer.append(raw_prediction)
        smoothed = int(collections.Counter(self.pred_buffer).most_common(1)[0][0])

        self.state_durations[smoothed] += dt

        alarm_triggered = False
        seconds_closed  = 0.0

        if smoothed in DROWSY_STATES:
            if self.drowsy_start_time is None:
                self.drowsy_start_time = now
            seconds_closed = now - self.drowsy_start_time

            # Determine appropriate alarm level
            target_level = self._desired_level(seconds_closed)

            if target_level > self.current_level:
                self._trigger_alarm(target_level)
                self.current_level = target_level
                alarm_triggered = True

        else:
            # Eyes open / yawning — reset closure timer
            if self.current_level > LEVEL_NONE:
                self._stop_alarm()
            self.drowsy_start_time = None
            self.current_level     = LEVEL_NONE

        return smoothed, alarm_triggered or (self.current_level > LEVEL_NONE), seconds_closed

    def _desired_level(self, seconds_closed):
        """Return which alarm level should be active right now."""
        if seconds_closed < self.alarm_threshold:
            return LEVEL_NONE

        if seconds_closed >= POWER_SECS:
            return LEVEL_POWER

        # Between ALARM_SECS and POWER_SECS — decide NORMAL vs SHORT
        now = time.time()
        recent = sum(1 for t in self.alarm_history
                     if now - t < REPEAT_WINDOW)
        return LEVEL_SHORT if recent >= REPEAT_MIN else LEVEL_NORMAL

    def _trigger_alarm(self, level):
        self._stop_alarm()  # stop whatever is playing

        label = {LEVEL_NORMAL: 'NORMAL', LEVEL_SHORT: 'SHORT', LEVEL_POWER: 'POWER'}[level]
        print(f"[ALARM] Level={label}  alarm #{self.alarm_count + 1}")

        if self._audio_ok:
            loops = -1 if level == LEVEL_POWER else (2 if level == LEVEL_SHORT else 0)
            self._sounds[level].play(loops=loops)

        self.alarm_history.append(time.time())
        self.alarm_count += 1

    def _stop_alarm(self):
        if self._audio_ok:
            pygame.mixer.stop()

    # ── Misc ─────────────────────────────────────────────────────────────────
    def reset(self):
        self.pred_buffer.clear()
        self.drowsy_start_time = None
        self.current_level     = LEVEL_NONE
        self._stop_alarm()

    def get_stats(self):
        total = sum(self.state_durations.values()) or 1
        return {
            'frames_processed': self.frame_count,
            'alarms_triggered': self.alarm_count,
            'state_percentages': {
                STATE_NAMES[k]: round(v / total * 100, 1)
                for k, v in self.state_durations.items()
                if k in STATE_NAMES
            },
        }

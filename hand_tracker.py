# ============================================================
# hand_tracker.py — Index-finger tap detection for click
# ============================================================
#
# Gesture definitions (same webcam as face):
#   - Single tap  = index finger moves DOWN then UP once
#   - Double tap  = two taps within DOUBLE_TAP_WINDOW_MS
#
# State machine:
#   IDLE  →  finger detected, record start Y
#   DOWN  →  finger tip dropped by >= TAP_DOWN_THRESHOLD
#   TAP   →  finger tip rose back by >= TAP_UP_THRESHOLD → tap counted
#   COOLDOWN → wait before next detection
# ============================================================

import time
import cv2
import numpy as np
from mediapipe.python.solutions import hands as mp_hands

from config import (
    TAP_DOWN_THRESHOLD,
    TAP_UP_THRESHOLD,
    TAP_MAX_DURATION_MS,
    DOUBLE_TAP_WINDOW_MS,
    TAP_COOLDOWN_MS,
    HAND_DETECT_EVERY_N_FRAMES,
)


# Landmark indices (MediaPipe Hands)
INDEX_TIP = 8
INDEX_MCP = 5
WRIST = 0


class HandTracker:

    # States
    IDLE = "idle"
    TRACKING_DOWN = "tracking_down"
    TRACKING_UP = "tracking_up"
    COOLDOWN = "cooldown"

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        self.state = self.IDLE

        # Y position of index tip when tracking started
        self._start_y = 0.0
        self._lowest_y = 0.0         # lowest Y reached during down motion
        self._motion_start_time = 0.0

        # Tap history for double-tap detection
        self._last_tap_time = 0.0
        self._tap_count = 0           # taps accumulated in current window

        # Cooldown timer
        self._cooldown_until = 0.0

        # Frame skip counter
        self._frame_counter = 0
        self._last_result = None
        self.index_finger_present = False

    def is_index_finger_present(self):
        """Check if the index finger is currently visible and extended."""
        return self.index_finger_present

    def process(self, frame):
        """Process a frame and return gesture dict or None.

        Returns:
            None                 — no gesture this frame
            {"gesture": "click"}       — single click
            {"gesture": "doubleclick"} — double click
        """

        # Skip frames for CPU efficiency
        self._frame_counter += 1
        if self._frame_counter % HAND_DETECT_EVERY_N_FRAMES != 0:
            return None

        now_ms = time.time() * 1000.0

        # Cooldown check
        if self.state == self.COOLDOWN:
            if now_ms >= self._cooldown_until:
                self.state = self.IDLE
            else:
                return None

        # Run MediaPipe Hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            # No hand visible — reset
            self.index_finger_present = False
            if self.state != self.COOLDOWN:
                self._reset_tracking()
            return None

        hand = results.multi_hand_landmarks[0]
        index_tip_y = hand.landmark[INDEX_TIP].y
        index_mcp_y = hand.landmark[INDEX_MCP].y
        wrist_y = hand.landmark[WRIST].y

        # Only track when index finger is extended (tip above MCP)
        finger_extended = index_tip_y < index_mcp_y
        self.index_finger_present = finger_extended

        if not finger_extended:
            if self.state not in (self.COOLDOWN,):
                self._reset_tracking()
            return None

        # ==========================================
        # STATE MACHINE
        # ==========================================

        if self.state == self.IDLE:
            # Finger just appeared / extended — record start position
            self._start_y = index_tip_y
            self._lowest_y = index_tip_y
            self._motion_start_time = now_ms
            self.state = self.TRACKING_DOWN
            return None

        elif self.state == self.TRACKING_DOWN:
            # Check timeout
            if now_ms - self._motion_start_time > TAP_MAX_DURATION_MS:
                self._reset_tracking()
                return None

            # Track the lowest point
            if index_tip_y > self._lowest_y:
                self._lowest_y = index_tip_y

            # Has finger dropped enough?
            drop = self._lowest_y - self._start_y
            if drop >= TAP_DOWN_THRESHOLD:
                self.state = self.TRACKING_UP
            return None

        elif self.state == self.TRACKING_UP:
            # Check timeout
            if now_ms - self._motion_start_time > TAP_MAX_DURATION_MS:
                self._reset_tracking()
                return None

            # Has finger risen back up?
            rise = self._lowest_y - index_tip_y
            if rise >= TAP_UP_THRESHOLD:
                # TAP completed!
                return self._register_tap(now_ms)

            return None

        return None

    def _register_tap(self, now_ms):
        """A single tap motion completed — check for double-tap."""
        elapsed_since_last = now_ms - self._last_tap_time

        if elapsed_since_last < DOUBLE_TAP_WINDOW_MS and self._tap_count >= 1:
            # Second tap within window → double click
            self._tap_count = 0
            self._last_tap_time = 0.0
            self._enter_cooldown(now_ms)
            return {"gesture": "doubleclick"}
        else:
            # First tap — wait to see if a second follows
            self._tap_count = 1
            self._last_tap_time = now_ms

            # Reset state to look for next tap immediately
            self.state = self.IDLE
            return None

    def check_pending_tap(self):
        """Call this each frame AFTER process() to check if a single
        tap should fire (no second tap arrived within window).

        Returns:
            None                 — nothing pending
            {"gesture": "click"} — single click fires
        """
        if self._tap_count < 1:
            return None

        now_ms = time.time() * 1000.0
        elapsed = now_ms - self._last_tap_time

        if elapsed >= DOUBLE_TAP_WINDOW_MS:
            # Window expired — fire single click
            self._tap_count = 0
            self._last_tap_time = 0.0
            self._enter_cooldown(now_ms)
            return {"gesture": "click"}

        return None

    def _enter_cooldown(self, now_ms):
        self.state = self.COOLDOWN
        self._cooldown_until = now_ms + TAP_COOLDOWN_MS

    def _reset_tracking(self):
        self.state = self.IDLE
        self._start_y = 0.0
        self._lowest_y = 0.0
        self._motion_start_time = 0.0

    def draw_debug(self, frame, label=None):
        """Draw hand tracking status on frame."""
        state_color = {
            self.IDLE: (180, 180, 180),
            self.TRACKING_DOWN: (0, 200, 255),
            self.TRACKING_UP: (0, 255, 200),
            self.COOLDOWN: (100, 100, 255),
        }
        color = state_color.get(self.state, (255, 255, 255))

        cv2.putText(
            frame,
            f"Hand: {self.state}",
            (30, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        if self._tap_count > 0:
            cv2.putText(
                frame,
                f"Taps: {self._tap_count} (waiting...)",
                (30, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        if label:
            cv2.putText(
                frame,
                label,
                (30, 490),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

"""Hand gesture tracking for gaze clutch and mouse clicks.

Gestures:
  - No index finger: gaze cursor paused.
  - Index finger extended: gaze cursor active.
  - Index-only down/up tap: immediate left click.
  - Index + middle raised, ring + pinky folded: one right click.
"""

import time

import cv2
from mediapipe.python.solutions import hands as mp_hands

from config import (
    HAND_DETECT_EVERY_N_FRAMES,
    TAP_COOLDOWN_MS,
    TAP_DOWN_THRESHOLD,
    TAP_MAX_DURATION_MS,
    TAP_UP_THRESHOLD,
)


INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_MCP = 9
RING_TIP = 16
RING_MCP = 13
PINKY_TIP = 20
PINKY_MCP = 17


class HandTracker:
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
        self._start_y = 0.0
        self._lowest_y = 0.0
        self._motion_start_time = 0.0
        self._cooldown_until = 0.0

        self._frame_counter = 0
        self.index_finger_present = False
        self.two_finger_present = False
        self._two_finger_latched = False

    def is_index_finger_present(self):
        """Whether the index finger currently enables gaze movement."""
        return self.index_finger_present

    @staticmethod
    def _extended(hand, tip_id, mcp_id):
        return hand.landmark[tip_id].y < hand.landmark[mcp_id].y

    def process(self, frame):
        """Process a frame and return a click gesture dictionary or None."""
        self._frame_counter += 1
        if self._frame_counter % HAND_DETECT_EVERY_N_FRAMES != 0:
            return None

        now_ms = time.time() * 1000.0
        if self.state == self.COOLDOWN:
            if now_ms >= self._cooldown_until:
                self.state = self.IDLE
            else:
                return None

        # Landmarks are normalized, so downscaling keeps gesture geometry.
        small_frame = cv2.resize(frame, (640, 360))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            self.index_finger_present = False
            self.two_finger_present = False
            self._two_finger_latched = False
            self._reset_tracking()
            return None

        hand = results.multi_hand_landmarks[0]
        index_extended = self._extended(hand, INDEX_TIP, INDEX_MCP)
        middle_extended = self._extended(hand, MIDDLE_TIP, MIDDLE_MCP)
        ring_extended = self._extended(hand, RING_TIP, RING_MCP)
        pinky_extended = self._extended(hand, PINKY_TIP, PINKY_MCP)

        # Ring and pinky must be folded so an open palm cannot right-click.
        two_finger_pose = (
            index_extended
            and middle_extended
            and not ring_extended
            and not pinky_extended
        )
        self.two_finger_present = two_finger_pose
        # Freeze gaze movement while changing into the click pose.
        self.index_finger_present = index_extended and not two_finger_pose

        if two_finger_pose:
            self._reset_tracking()
            if not self._two_finger_latched:
                self._two_finger_latched = True
                self._enter_cooldown(now_ms)
                return {"gesture": "rightclick"}
            return None

        # Releasing the pose rearms it for a future right click.
        self._two_finger_latched = False

        # Only an index-only pose participates in tap detection. Other poses
        # may retain the clutch but cannot accidentally become a tap.
        if not index_extended or middle_extended:
            self._reset_tracking()
            return None

        index_tip_y = hand.landmark[INDEX_TIP].y

        if self.state == self.IDLE:
            self._start_y = index_tip_y
            self._lowest_y = index_tip_y
            self._motion_start_time = now_ms
            self.state = self.TRACKING_DOWN
            return None

        if self.state == self.TRACKING_DOWN:
            if now_ms - self._motion_start_time > TAP_MAX_DURATION_MS:
                self._reset_tracking()
                return None

            self._lowest_y = max(self._lowest_y, index_tip_y)
            if self._lowest_y - self._start_y >= TAP_DOWN_THRESHOLD:
                self.state = self.TRACKING_UP
            return None

        if self.state == self.TRACKING_UP:
            if now_ms - self._motion_start_time > TAP_MAX_DURATION_MS:
                self._reset_tracking()
                return None

            if self._lowest_y - index_tip_y >= TAP_UP_THRESHOLD:
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
            2,
        )

        if self.two_finger_present:
            cv2.putText(
                frame,
                "INDEX + MIDDLE: RIGHT CLICK",
                (30, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

        if label:
            cv2.putText(
                frame,
                label,
                (30, 490),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

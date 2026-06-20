# ============================================================
# filters.py — Smoothing filters for cursor position
# ============================================================

from collections import deque
import numpy as np

from config import (
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
    EDGE_ZONE_PX,
    EDGE_ALPHA,
)


# ============================================================
# Moving Average 2D  (kept for simple usage)
# ============================================================
class MovingAverage2D:
    def __init__(self, size=8):
        self.x_values = deque(maxlen=size)
        self.y_values = deque(maxlen=size)

    def update(self, x, y):
        self.x_values.append(float(x))
        self.y_values.append(float(y))

        avg_x = sum(self.x_values) / len(self.x_values)
        avg_y = sum(self.y_values) / len(self.y_values)

        return avg_x, avg_y

    def reset(self):
        self.x_values.clear()
        self.y_values.clear()


# ============================================================
# Exponential Moving Average 2D
# ============================================================
class EMA2D:
    def __init__(self, alpha=0.18):
        self.alpha = float(alpha)
        self.initialized = False
        self.x = 0.0
        self.y = 0.0

    def update(self, x, y):
        x = float(x)
        y = float(y)

        if not self.initialized:
            self.x = x
            self.y = y
            self.initialized = True
            return self.x, self.y

        self.x = self.alpha * x + (1.0 - self.alpha) * self.x
        self.y = self.alpha * y + (1.0 - self.alpha) * self.y

        return self.x, self.y

    def reset(self):
        self.initialized = False
        self.x = 0.0
        self.y = 0.0


# ============================================================
# Kalman Filter 2D  (constant-velocity model)
# ============================================================
#
# State vector: [x, y, vx, vy]
# Measurement:  [x, y]
#
# This is dramatically better than MA+EMA because it:
#   - predicts based on velocity (low lag for saccades)
#   - smooths noise during fixation
#   - adapts automatically via covariance updates
# ============================================================
class KalmanFilter2D:
    def __init__(
        self,
        process_noise=KALMAN_PROCESS_NOISE,
        measurement_noise=KALMAN_MEASUREMENT_NOISE,
    ):
        # State vector [x, y, vx, vy]
        self.x = np.zeros(4)

        # State transition  (assume dt=1 per frame)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance
        self.Q = np.eye(4) * process_noise

        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise

        # Estimate covariance
        self.P = np.eye(4) * 1.0

        self.initialized = False

    def update(self, mx, my):
        """Feed a new measurement (mx, my) and return smoothed (x, y)."""
        mx = float(mx)
        my = float(my)

        if not self.initialized:
            self.x = np.array([mx, my, 0.0, 0.0])
            self.initialized = True
            return mx, my

        # --- Predict ---
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # --- Update ---
        z = np.array([mx, my])
        y = z - self.H @ x_pred                        # innovation
        S = self.H @ P_pred @ self.H.T + self.R        # innovation cov
        K = P_pred @ self.H.T @ np.linalg.inv(S)       # Kalman gain

        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        return float(self.x[0]), float(self.x[1])

    def reset(self):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1.0
        self.initialized = False


# ============================================================
# Adaptive EMA 2D  — alpha varies with velocity
# ============================================================
# Fast movement → high alpha (responsive)
# Fixation       → low alpha (stable)
# ============================================================
class AdaptiveEMA2D:
    def __init__(self, alpha_low=0.08, alpha_high=0.35, speed_threshold=60.0):
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.speed_threshold = speed_threshold
        self.initialized = False
        self.x = 0.0
        self.y = 0.0

    def update(self, x, y):
        x = float(x)
        y = float(y)

        if not self.initialized:
            self.x = x
            self.y = y
            self.initialized = True
            return self.x, self.y

        dx = x - self.x
        dy = y - self.y
        speed = (dx * dx + dy * dy) ** 0.5

        # Lerp alpha based on speed
        t = min(speed / max(self.speed_threshold, 1e-6), 1.0)
        alpha = self.alpha_low + (self.alpha_high - self.alpha_low) * t

        self.x = alpha * x + (1.0 - alpha) * self.x
        self.y = alpha * y + (1.0 - alpha) * self.y

        return self.x, self.y

    def reset(self):
        self.initialized = False
        self.x = 0.0
        self.y = 0.0


# ============================================================
# Edge Damper — extra smoothing near screen borders
# ============================================================
# Gaze predictions near the edges tend to jitter because the
# polynomial model is extrapolating.  This applies a stronger
# EMA alpha when the cursor is within EDGE_ZONE_PX of any
# screen border.
# ============================================================
class EdgeDamper:
    def __init__(
        self,
        screen_w,
        screen_h,
        edge_zone=EDGE_ZONE_PX,
        edge_alpha=EDGE_ALPHA,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.edge_zone = edge_zone
        self.edge_alpha = edge_alpha
        self.x = None
        self.y = None

    def update(self, x, y):
        x = float(x)
        y = float(y)

        if self.x is None:
            self.x = x
            self.y = y
            return x, y

        # How far from nearest edge?
        dx_edge = min(x, self.screen_w - x)
        dy_edge = min(y, self.screen_h - y)
        nearest = min(dx_edge, dy_edge)

        if nearest < self.edge_zone:
            # Inside edge zone → damp
            self.x = self.edge_alpha * x + (1.0 - self.edge_alpha) * self.x
            self.y = self.edge_alpha * y + (1.0 - self.edge_alpha) * self.y
        else:
            # Outside edge zone → pass through
            self.x = x
            self.y = y

        return self.x, self.y

    def reset(self):
        self.x = None
        self.y = None
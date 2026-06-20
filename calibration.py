# ============================================================
# calibration.py — Polynomial regression calibration system
# ============================================================

import numpy as np

from config import (
    CALIBRATION_SCHEMA_VERSION,
    RIDGE_LAMBDA,
)


class GazeCalibration:

    def __init__(self, targets=None):

        # Calibration targets ordered: center first (easiest), then
        # middle row, inner grid, top edge, bottom edge.
        # Starting at center makes the first point easy and allows
        # the reference eye height to be captured immediately.
        self.targets = targets or [

            # Center (first — easiest point, captures ref eye height)
            (0.50, 0.50),

            # Middle row
            (0.03, 0.50),
            (0.25, 0.50),
            (0.75, 0.50),
            (0.97, 0.50),

            # Inner grid point
            (0.35, 0.30),
            (0.65, 0.70),

            # Top row
            (0.03, 0.05),
            (0.25, 0.05),
            (0.50, 0.05),
            (0.75, 0.05),
            (0.97, 0.05),

            # Bottom row
            (0.03, 0.95),
            (0.25, 0.95),
            (0.50, 0.95),
            (0.75, 0.95),
            (0.97, 0.95),
        ]

        self.samples = []

        self.current_index = 0

        self.coeff_x = None
        self.coeff_y = None

        # Mean IPD at calibration time (for distance compensation)
        self.mean_ipd = None

    # ==========================================
    # TARGET
    # ==========================================
    def current_target(self):

        if self.is_complete():
            return None

        return self.targets[self.current_index]

    # ==========================================
    # STABILITY CHECK
    # ==========================================
    def stable(
        self,
        samples,
        threshold=0.015
    ):

        if len(samples) < 5:
            return False

        xs = [s[0] for s in samples]
        ys = [s[1] for s in samples]

        std_x = np.std(xs)
        std_y = np.std(ys)

        return (
            std_x < threshold
            and
            std_y < threshold
        )

    # ==========================================
    # ADD SAMPLE
    # ==========================================
    def add_sample(
        self,
        features
    ):

        tx, ty = self.targets[
            self.current_index
        ]

        self.samples.append({

            # ======================================
            # RAW GAZE
            # ======================================
            "gaze_x": float(features["gaze_x"]),
            "gaze_y": float(features["gaze_y"]),

            # ======================================
            # HEAD-COMPENSATED GAZE (primary signal)
            # ======================================
            "comp_gx": float(features["comp_gx"]),
            "comp_gy": float(features["comp_gy"]),

            # ======================================
            # LEFT EYE
            # ======================================
            "left_gx": float(features["left_gx"]),
            "left_gy": float(features["left_gy"]),
            "left_open": float(features["left_open"]),

            # ======================================
            # RIGHT EYE
            # ======================================
            "right_gx": float(features["right_gx"]),
            "right_gy": float(features["right_gy"]),
            "right_open": float(features["right_open"]),

            # ======================================
            # HEAD
            # ======================================
            "yaw": float(features["yaw"]),
            "pitch": float(features["pitch"]),

            # ======================================
            # GEOMETRY
            # ======================================
            "ipd_px": float(features.get("ipd_px", 0.0)),

            # ======================================
            # TARGET
            # ======================================
            "target_x": float(tx),
            "target_y": float(ty),
        })

        self.current_index += 1

    # ==========================================
    # COMPLETE
    # ==========================================
    def is_complete(self):

        return (
            self.current_index >=
            len(self.targets)
        )

    # ==========================================
    # FEATURE VECTOR (expanded basis)
    # ==========================================
    def _basis(
        self,
        sample
    ):
        # Head-compensated gaze is the PRIMARY signal
        cgx = sample.get("comp_gx", sample["gaze_x"])
        cgy = sample.get("comp_gy", sample["gaze_y"])

        # Raw gaze and head pose as secondary correction terms
        gx = sample["gaze_x"]
        gy = sample["gaze_y"]
        yaw = sample["yaw"]
        pitch = sample["pitch"]

        # IPD ratio: current / calibration mean
        ipd = sample.get("ipd_px", 0.0)
        if self.mean_ipd and self.mean_ipd > 1e-3:
            ipd_ratio = ipd / self.mean_ipd
        else:
            ipd_ratio = 1.0

        return np.array([

            1.0,

            # ---- Primary: head-compensated gaze ----
            cgx,
            cgy,

            # Quadratic compensated
            cgx ** 2,
            cgy ** 2,
            cgx * cgy,

            # ---- Secondary: raw iris position ----
            gx,
            gy,

            # ---- Head pose (residual correction) ----
            yaw,
            pitch,

            # ---- Cross terms ----
            yaw * cgx,
            pitch * cgy,

            # ---- Eye openness (squint correction) ----
            sample["left_open"],
            sample["right_open"],

            # ---- IPD distance ratio ----
            ipd_ratio,

        ])

    # ==========================================
    # FIT (ridge regression)
    # ==========================================
    def fit(self):

        # Compute mean IPD from samples for runtime normalisation
        ipd_values = [s.get("ipd_px", 0.0) for s in self.samples]
        ipd_values = [v for v in ipd_values if v > 1.0]
        if ipd_values:
            self.mean_ipd = float(sum(ipd_values) / len(ipd_values))
        else:
            self.mean_ipd = None

        X = np.array([
            self._basis(s)
            for s in self.samples
        ])

        yx = np.array([
            s["target_x"]
            for s in self.samples
        ])

        yy = np.array([
            s["target_y"]
            for s in self.samples
        ])

        # Ridge regression (Tikhonov regularisation)
        # Solves: (X^T X + λI) β = X^T y
        # This prevents overfitting on noisy webcam data, especially
        # near screen edges where the polynomial would otherwise
        # extrapolate wildly.
        n_features = X.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0.0  # don't regularise the intercept

        XtX = X.T @ X + RIDGE_LAMBDA * I
        Xty_x = X.T @ yx
        Xty_y = X.T @ yy

        self.coeff_x = np.linalg.solve(XtX, Xty_x)
        self.coeff_y = np.linalg.solve(XtX, Xty_y)

    # ==========================================
    # PREDICT
    # ==========================================
    def predict(
        self,
        features
    ):

        b = self._basis(features)

        x = float(np.dot(
            b,
            self.coeff_x
        ))

        y = float(np.dot(
            b,
            self.coeff_y
        ))

        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

        return x, y

    # ==========================================
    # VALIDATION
    # ==========================================
    def validation_targets(self):

        return [

            (0.15, 0.15),
            (0.85, 0.15),

            (0.15, 0.85),
            (0.85, 0.85),

            (0.50, 0.20),
            (0.50, 0.80),
        ]

    # ==========================================
    # ERROR
    # ==========================================
    def compute_error(
        self,
        predicted,
        target,
        screen_w,
        screen_h
    ):

        px = predicted[0] * screen_w
        py = predicted[1] * screen_h

        tx = target[0] * screen_w
        ty = target[1] * screen_h

        return float(np.sqrt(
            (px - tx) ** 2 +
            (py - ty) ** 2
        ))

    # ==========================================
    # QUALITY
    # ==========================================
    def score_quality(
        self,
        avg_error
    ):

        if avg_error < 80:
            return "EXCELLENT"

        elif avg_error < 150:
            return "GOOD"

        elif avg_error < 250:
            return "AVERAGE"

        else:
            return "POOR"

    # ==========================================
    # SAVE
    # ==========================================
    def to_dict(self):

        return {

            "version": CALIBRATION_SCHEMA_VERSION,

            "targets": self.targets,

            "samples": self.samples,

            "coeff_x": self.coeff_x.tolist(),

            "coeff_y": self.coeff_y.tolist(),

            "mean_ipd": self.mean_ipd,
        }

    # ==========================================
    # LOAD
    # ==========================================
    @classmethod
    def from_dict(
        cls,
        data
    ):
        version = data.get("version", 0)
        if version < CALIBRATION_SCHEMA_VERSION:
            print(
                f"[WARN] Profile version {version} is older than "
                f"current {CALIBRATION_SCHEMA_VERSION}. "
                f"Recalibration recommended for best accuracy."
            )

        obj = cls(
            targets=data.get("targets")
        )

        obj.samples = data.get(
            "samples",
            []
        )

        obj.current_index = len(
            obj.samples
        )

        obj.coeff_x = np.array(
            data["coeff_x"]
        )

        obj.coeff_y = np.array(
            data["coeff_y"]
        )

        obj.mean_ipd = data.get("mean_ipd", None)

        return obj
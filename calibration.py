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

        # Neutral head pose baseline (for serialization)
        self.neutral_yaw = 0.0
        self.neutral_pitch = 0.0

        # Runtime normalization captured at the first center point.
        self.reference_eye_height = None

        # Standardization for the compact regression feature vector.
        self.feature_mean = None
        self.feature_scale = None

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
            # HEAD-COMPENSATED GAZE (calculated relative to baseline)
            # ======================================
            "comp_gx": float(features.get("comp_gx", features["gaze_x"])),
            "comp_gy": float(features.get("comp_gy", features["gaze_y"])),

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
    # FEATURE VECTOR (compact basis)
    # ==========================================
    def _raw_basis(
        self,
        sample
    ):
        gx = sample["gaze_x"]
        gy = sample["gaze_y"]
        yaw = sample["yaw"] - self.neutral_yaw
        pitch = sample["pitch"] - self.neutral_pitch

        return np.array([
            gx,
            gy,
            yaw,
            pitch,
            gx * gx,
            gy * gy,
            gx * gy,
        ], dtype=np.float64)

    def _basis(self, sample):
        raw = self._raw_basis(sample)
        if self.feature_mean is None or self.feature_scale is None:
            normalized = raw
        else:
            normalized = (raw - self.feature_mean) / self.feature_scale
        return np.concatenate(([1.0], normalized))

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

        raw_X = np.array([self._raw_basis(s) for s in self.samples])
        self.feature_mean = np.mean(raw_X, axis=0)
        self.feature_scale = np.std(raw_X, axis=0)
        self.feature_scale[self.feature_scale < 1e-4] = 1.0
        X = np.column_stack((
            np.ones(len(raw_X)),
            (raw_X - self.feature_mean) / self.feature_scale,
        ))

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

            "neutral_yaw": self.neutral_yaw,

            "neutral_pitch": self.neutral_pitch,

            "reference_eye_height": self.reference_eye_height,

            "feature_mean": self.feature_mean.tolist(),

            "feature_scale": self.feature_scale.tolist(),
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

        obj.mean_ipd = data.get("mean_ipd", None)

        obj.neutral_yaw = data.get("neutral_yaw", 0.0)
        obj.neutral_pitch = data.get("neutral_pitch", 0.0)

        obj.reference_eye_height = data.get("reference_eye_height", None)

        if version < CALIBRATION_SCHEMA_VERSION:
            # Older coefficients used a different, over-parameterized basis.
            # Refit from the saved calibration samples using the current model.
            obj.fit()
        else:
            obj.feature_mean = np.array(data["feature_mean"], dtype=np.float64)
            obj.feature_scale = np.array(data["feature_scale"], dtype=np.float64)
            obj.coeff_x = np.array(data["coeff_x"], dtype=np.float64)
            obj.coeff_y = np.array(data["coeff_y"], dtype=np.float64)

        return obj

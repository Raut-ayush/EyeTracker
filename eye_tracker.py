# ============================================================
# eye_tracker.py — Extracts gaze / head features from webcam
# ============================================================

import cv2
import numpy as np
from mediapipe.python.solutions import face_mesh

from config import (
    BLINK_OPENNESS_THRESHOLD,
    EYE_MODE,
    HEAD_COUPLING_X,
    HEAD_COUPLING_Y,
)


class EyeTracker:

    def __init__(self):

        self.face_mesh = face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        # ==========================================
        # LEFT EYE
        # ==========================================
        self.LEFT_EYE_IRIS = [469, 470, 471, 472]

        self.LEFT_EYE_OUTER = 33
        self.LEFT_EYE_INNER = 133

        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145

        # ==========================================
        # RIGHT EYE
        # ==========================================
        self.RIGHT_EYE_IRIS = [474, 475, 476, 477]

        self.RIGHT_EYE_OUTER = 362
        self.RIGHT_EYE_INNER = 263

        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374

        # ==========================================
        # FACE
        # ==========================================
        self.NOSE_TIP = 1

        self.LEFT_FACE = 234
        self.RIGHT_FACE = 454

        self.FOREHEAD = 10
        self.CHIN = 152

        # Reference eye height captured during calibration
        self._ref_eye_h = None

        # Neutral head pose baseline (captured before calibration)
        self._neutral_yaw = 0.0
        self._neutral_pitch = 0.0

    # ==========================================
    # SET REFERENCE EYE HEIGHT
    # ==========================================
    def set_reference_eye_height(self, ref_h):
        """Store reference eye height for vertical normalisation."""
        self._ref_eye_h = ref_h

    def set_neutral_pose(self, yaw, pitch):
        """Store neutral head pose baseline for delta computation."""
        self._neutral_yaw = float(yaw)
        self._neutral_pitch = float(pitch)

    # ==========================================
    # HEAD-COMPENSATED GAZE
    # ==========================================
    def _head_compensated_gaze(self, gaze_x, gaze_y, yaw, pitch):
        """Combine iris position with head rotation for a wider signal.

        When you turn your head right, the iris appears to drift left
        (back toward centre) because the camera sees the eye from a
        different angle.  We ADD the head rotation delta back to the
        raw iris position to undo this perspective shift.

        Returns (comp_gx, comp_gy) with a much wider dynamic range
        than the raw iris values alone.
        """
        delta_yaw = yaw - self._neutral_yaw
        delta_pitch = pitch - self._neutral_pitch

        comp_gx = gaze_x + delta_yaw * HEAD_COUPLING_X
        comp_gy = gaze_y + delta_pitch * HEAD_COUPLING_Y

        return float(comp_gx), float(comp_gy)

    # ==========================================
    # IRIS CENTER
    # ==========================================
    def _iris_center(
        self,
        landmarks,
        iris_ids
    ):

        xs = [
            landmarks.landmark[i].x
            for i in iris_ids
        ]

        ys = [
            landmarks.landmark[i].y
            for i in iris_ids
        ]

        return (
            float(np.mean(xs)),
            float(np.mean(ys))
        )

    # ==========================================
    # EYE FEATURES
    # ==========================================
    def _eye_features(
        self,
        landmarks,
        iris_ids,
        corner_a,
        corner_b,
        top_id,
        bottom_id
    ):

        iris_x, iris_y = self._iris_center(
            landmarks,
            iris_ids
        )

        x1 = landmarks.landmark[corner_a].x
        x2 = landmarks.landmark[corner_b].x

        y1 = landmarks.landmark[top_id].y
        y2 = landmarks.landmark[bottom_id].y

        left_x = min(x1, x2)
        right_x = max(x1, x2)

        top_y = min(y1, y2)
        bottom_y = max(y1, y2)

        eye_w = right_x - left_x
        eye_h = bottom_y - top_y

        if eye_w <= 1e-6 or eye_h <= 1e-6:
            return None

        # ==========================================
        # NORMALIZED IRIS POSITION
        # ==========================================
        gx = (iris_x - left_x) / eye_w
        gy = (iris_y - top_y) / eye_h

        # Adaptive vertical normalisation — if a reference
        # eye height was captured during calibration, scale
        # gy by the ratio of current height to reference to
        # reduce vertical drift from blinks and squinting.
        if self._ref_eye_h is not None and self._ref_eye_h > 1e-6:
            scale = eye_h / self._ref_eye_h
            # Only apply a mild correction (clamp scale)
            scale = float(np.clip(scale, 0.7, 1.4))
            gy = gy * scale

        gx = float(np.clip(gx, 0.0, 1.0))
        gy = float(np.clip(gy, 0.0, 1.0))

        # ==========================================
        # EYE OPENNESS
        # ==========================================
        openness = eye_h / eye_w

        return (
            gx,
            gy,
            openness,
            iris_x,
            iris_y,
            eye_h,
            (left_x, top_y),
            (right_x, bottom_y)
        )

    # ==========================================
    # HEAD POSE (improved)
    # ==========================================
    def _head_pose(
        self,
        landmarks
    ):
        """Estimate yaw and pitch using face geometry ratios.

        Yaw  → ratio of nose-to-left vs nose-to-right face width.
        Pitch → ratio of nose-to-forehead vs nose-to-chin distance.

        Returns (yaw, pitch, face_center_x, face_center_y).
        Positive yaw = looking right.  Positive pitch = looking down.
        """

        nose = landmarks.landmark[self.NOSE_TIP]

        left_face = landmarks.landmark[self.LEFT_FACE]
        right_face = landmarks.landmark[self.RIGHT_FACE]

        forehead = landmarks.landmark[self.FOREHEAD]
        chin = landmarks.landmark[self.CHIN]

        face_center_x = (
            left_face.x +
            right_face.x
        ) / 2.0

        face_center_y = (
            forehead.y +
            chin.y
        ) / 2.0

        # --- Improved yaw using ratio method ---
        dist_left = abs(nose.x - left_face.x)
        dist_right = abs(nose.x - right_face.x)
        total_w = dist_left + dist_right

        if total_w > 1e-6:
            # 0.5 = centred, <0.5 = left, >0.5 = right
            yaw = (dist_left / total_w) - 0.5
        else:
            yaw = 0.0

        # --- Improved pitch using ratio method ---
        dist_up = abs(nose.y - forehead.y)
        dist_down = abs(nose.y - chin.y)
        total_h = dist_up + dist_down

        if total_h > 1e-6:
            pitch = (dist_up / total_h) - 0.5
        else:
            pitch = 0.0

        return (
            float(yaw),
            float(pitch),
            float(face_center_x),
            float(face_center_y)
        )

    # ==========================================
    # INTER-PUPILLARY DISTANCE (pixels)
    # ==========================================
    def _ipd_pixels(
        self,
        landmarks,
        frame_w
    ):
        """Distance between left and right iris centers in pixels."""
        lx, ly = self._iris_center(landmarks, self.LEFT_EYE_IRIS)
        rx, ry = self._iris_center(landmarks, self.RIGHT_EYE_IRIS)

        dx = (lx - rx) * frame_w
        dy = (ly - ry) * frame_w   # intentionally use frame_w for aspect
        return float((dx * dx + dy * dy) ** 0.5)

    # ==========================================
    # MAIN
    # ==========================================
    def get_gaze_features(
        self,
        frame
    ):
        h_orig, w_orig = frame.shape[:2]

        # Downscale to 640x360 for 4x CPU speedup
        small_frame = cv2.resize(frame, (640, 360))

        rgb_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(
            rgb_frame
        )

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]

        h, w = h_orig, w_orig

        # ==========================================
        # LEFT EYE
        # ==========================================
        left_eye = self._eye_features(
            landmarks,
            self.LEFT_EYE_IRIS,
            self.LEFT_EYE_OUTER,
            self.LEFT_EYE_INNER,
            self.LEFT_EYE_TOP,
            self.LEFT_EYE_BOTTOM
        )

        # ==========================================
        # RIGHT EYE
        # ==========================================
        right_eye = self._eye_features(
            landmarks,
            self.RIGHT_EYE_IRIS,
            self.RIGHT_EYE_OUTER,
            self.RIGHT_EYE_INNER,
            self.RIGHT_EYE_TOP,
            self.RIGHT_EYE_BOTTOM
        )

        if left_eye is None or right_eye is None:
            return None

        (
            lgx,
            lgy,
            lopen,
            lix,
            liy,
            left_eye_h,
            lcorner1,
            lcorner2
        ) = left_eye

        (
            rgx,
            rgy,
            ropen,
            rix,
            riy,
            right_eye_h,
            rcorner1,
            rcorner2
        ) = right_eye

        # ==========================================
        # BLINK GUARD
        # ==========================================
        if lopen < BLINK_OPENNESS_THRESHOLD or ropen < BLINK_OPENNESS_THRESHOLD:
            return None  # skip blink frames entirely

        # ==========================================
        # HEAD POSE
        # ==========================================
        yaw, pitch, face_cx, face_cy = \
            self._head_pose(
                landmarks
            )

        # ==========================================
        # RAW GAZE (respects EYE_MODE setting)
        # ==========================================
        if EYE_MODE == "left":
            gaze_x = lgx
            gaze_y = lgy
        elif EYE_MODE == "right":
            gaze_x = rgx
            gaze_y = rgy
        else:  # "both"
            gaze_x = (lgx + rgx) / 2.0
            gaze_y = (lgy + rgy) / 2.0

        # ==========================================
        # IPD (pixels)
        # ==========================================
        ipd_px = self._ipd_pixels(landmarks, w)

        # ==========================================
        # AVERAGE EYE HEIGHT (for ref tracking)
        # ==========================================
        avg_eye_h = (left_eye_h + right_eye_h) / 2.0

        # ==========================================
        # PIXELS
        # ==========================================
        left_iris_px = (
            int(lix * w),
            int(liy * h)
        )

        right_iris_px = (
            int(rix * w),
            int(riy * h)
        )

        face_center_px = (
            int(face_cx * w),
            int(face_cy * h)
        )

        # ==========================================
        # HEAD-COMPENSATED GAZE
        # ==========================================
        comp_gx, comp_gy = self._head_compensated_gaze(
            gaze_x, gaze_y, yaw, pitch
        )

        return {

            # ======================================
            # MAIN
            # ======================================
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,

            # ======================================
            # HEAD-COMPENSATED (primary for regression)
            # ======================================
            "comp_gx": comp_gx,
            "comp_gy": comp_gy,

            # ======================================
            # LEFT EYE
            # ======================================
            "left_gx": lgx,
            "left_gy": lgy,
            "left_open": lopen,

            # ======================================
            # RIGHT EYE
            # ======================================
            "right_gx": rgx,
            "right_gy": rgy,
            "right_open": ropen,

            # ======================================
            # HEAD
            # ======================================
            "yaw": yaw,
            "pitch": pitch,

            # ======================================
            # GEOMETRY
            # ======================================
            "ipd_px": ipd_px,
            "avg_eye_h": avg_eye_h,

            # ======================================
            # DEBUG
            # ======================================
            "left_iris_px": left_iris_px,
            "right_iris_px": right_iris_px,

            "face_center_px": face_center_px,

            "left_eye_box": (
                (
                    int(lcorner1[0] * w),
                    int(lcorner1[1] * h)
                ),
                (
                    int(lcorner2[0] * w),
                    int(lcorner2[1] * h)
                ),
            ),

            "right_eye_box": (
                (
                    int(rcorner1[0] * w),
                    int(rcorner1[1] * h)
                ),
                (
                    int(rcorner2[0] * w),
                    int(rcorner2[1] * h)
                ),
            ),
        }

    # ==========================================
    # DEBUG
    # ==========================================
    def draw_debug(
        self,
        frame,
        features
    ):

        if features is None:
            return

        cv2.circle(
            frame,
            features["left_iris_px"],
            5,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            frame,
            features["right_iris_px"],
            5,
            (0, 255, 0),
            -1
        )

        (lx1, ly1), (lx2, ly2) = \
            features["left_eye_box"]

        (rx1, ry1), (rx2, ry2) = \
            features["right_eye_box"]

        cv2.rectangle(
            frame,
            (lx1, ly1),
            (lx2, ly2),
            (255, 0, 0),
            1
        )

        cv2.rectangle(
            frame,
            (rx1, ry1),
            (rx2, ry2),
            (255, 0, 0),
            1
        )

        cv2.circle(
            frame,
            features["face_center_px"],
            5,
            (0, 255, 255),
            -1
        )

        cv2.putText(
            frame,
            f"Yaw: {features['yaw']:.3f}",
            (30, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Pitch: {features['pitch']:.3f}",
            (30, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"IPD: {features['ipd_px']:.0f}px",
            (30, 390),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
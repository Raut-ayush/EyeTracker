# ============================================================
# config.py — Central constants for EyeTracker
# Edit values here to tune system behaviour without touching
# any other module.
# ============================================================

# ============================================================
# CAMERA
# ============================================================
CAMERA_INDEX = 0          # webcam index (0 = default, 1 = secondary)
CAMERA_WIDTH = 1280       # requested capture width  (actual may be lower)
CAMERA_HEIGHT = 720       # requested capture height

# ============================================================
# CURSOR MOVEMENT
# ============================================================
DEAD_ZONE = 5             # pixels of cursor movement to ignore (reduces jitter)
SCREEN_MARGIN = 15        # pixels from screen edge cursor is clamped to
PYAUTOGUI_PAUSE = 0.01    # PyAutoGUI's default 0.10s pause makes tracking sluggish
GAZE_MEDIAN_WINDOW = 5    # rejects brief involuntary gaze spikes (must be odd)
CURSOR_ALPHA_LOW = 0.025  # fixation response: lower = steadier cursor
CURSOR_ALPHA_HIGH = 0.30  # large gaze-shift response
CURSOR_SPEED_THRESHOLD = 150.0  # pixels needed to approach fast response

# ============================================================
# BLINK GUARD
# ============================================================
# If either eye openness ratio drops below this value the
# gaze update is skipped (prevents blink-induced cursor jumps).
BLINK_OPENNESS_THRESHOLD = 0.17

# ============================================================
# KALMAN FILTER
# ============================================================
KALMAN_PROCESS_NOISE = 8e-4   # lower = smoother but laggier
KALMAN_MEASUREMENT_NOISE = 0.06  # higher = trusts sensor less

# ============================================================
# EDGE DAMPING
# ============================================================
# Within this many pixels of the screen border, apply extra EMA.
EDGE_ZONE_PX = 100
EDGE_ALPHA = 0.10          # alpha used inside edge zone (lower = more damp)

# ============================================================
# CALIBRATION
# ============================================================
CALIBRATION_SAMPLES = 30       # frames captured per calibration point
CALIBRATION_TIMEOUT = 10.0     # seconds max wait per point
CALIBRATION_SETTLE_SECONDS = 0.55  # ignore the initial eye saccade at each point
CALIBRATION_STABILITY_WINDOW = 8   # rolling frames used to detect real motion
HEAD_STABILITY_YAW = 0.008     # rolling yaw standard deviation limit
HEAD_STABILITY_PITCH = 0.008   # rolling pitch standard deviation limit
GAZE_STABILITY_X = 0.018       # rolling horizontal iris-position std limit
GAZE_STABILITY_Y = 0.022       # rolling vertical iris-position std limit
CALIBRATION_SCHEMA_VERSION = 9 # increment when profile format/model changes
RIDGE_LAMBDA = 1.0             # regularisation for the standardized compact model

# ============================================================
# HAND TRACKER / TAP GESTURE
# ============================================================
# A tap = index finger tip moves DOWN by >= TAP_DOWN_THRESHOLD
#          then back UP by >= TAP_UP_THRESHOLD (normalised 0-1 frame coords).
TAP_DOWN_THRESHOLD = 0.045    # normalised Y drop to start a tap
TAP_UP_THRESHOLD   = 0.030    # normalised Y rise to complete the tap

# Max frames a single tap motion can span before it is cancelled.
TAP_MAX_DURATION_MS = 600     # milliseconds

# If a second complete tap lands within this window → double click.
DOUBLE_TAP_WINDOW_MS = 500    # milliseconds

# Cooldown after any click fires (prevents repeat triggers).
TAP_COOLDOWN_MS = 350         # milliseconds

# Run hand detection every N frames (1 = every frame, 2 = every other, …)
# Increase if CPU is overloaded.
HAND_DETECT_EVERY_N_FRAMES = 2

# ============================================================
# HEAD-GAZE COUPLING (additive head compensation)
# ============================================================
# These gains control how much head rotation contributes to the
# computed gaze direction.  The idea: when you turn your head
# right the iris appears to shift left (back toward centre),
# so we ADD the head rotation back to the iris position to
# "undo" the perspective shift.
#
# Higher values = head movement has more influence on cursor.
# Typical range: 2.0 – 6.0.  Start conservative and tune up.
HEAD_COUPLING_X = 3.5         # yaw  → horizontal gaze contribution
HEAD_COUPLING_Y = 4.0         # pitch → vertical gaze contribution

# ============================================================
# NEUTRAL POSE CAPTURE
# ============================================================
NEUTRAL_POSE_FRAMES = 40      # frames to average for neutral pose

# ============================================================
# EYE DOMINANCE  (future use — both eyes used by default)
# ============================================================
# "both"   → average left + right
# "left"   → left eye only
# "right"  → right eye only
EYE_MODE = "both"

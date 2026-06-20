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
HEAD_STABILITY_YAW = 0.12      # max yaw  deviation to count as "stable"
HEAD_STABILITY_PITCH = 0.12    # max pitch deviation to count as "stable"
CALIBRATION_SCHEMA_VERSION = 8 # increment when profile format changes
RIDGE_LAMBDA = 5e-4            # Tikhonov regularisation strength

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

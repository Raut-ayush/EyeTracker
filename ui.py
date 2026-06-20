# ============================================================
# ui.py — OpenCV-based UI rendering
# ============================================================

import cv2
import numpy as np
import math


def _paste_preview(canvas, preview, x, y, max_w=360, max_h=240):
    ph, pw = preview.shape[:2]
    scale = min(max_w / pw, max_h / ph, 1.0)
    new_w = max(1, int(pw * scale))
    new_h = max(1, int(ph * scale))

    small = cv2.resize(preview, (new_w, new_h))
    h, w = canvas.shape[:2]

    x2 = min(w, x + new_w)
    y2 = min(h, y + new_h)

    canvas[y:y2, x:x2] = small[:y2 - y, :x2 - x]


# ============================================================
# MAIN MENU
# ============================================================
def render_main_menu_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Dark gradient background
    for row in range(720):
        t = row / 720.0
        b = int(15 + 25 * t)
        g = int(10 + 15 * t)
        r = int(20 + 30 * t)
        frame[row, :] = (b, g, r)

    # Title
    cv2.putText(frame, "EyeTracker AI Mouse", (340, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (100, 255, 200), 3)

    # Subtitle
    cv2.putText(frame, "Gaze + Gesture Control System", (380, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

    # Divider line
    cv2.line(frame, (380, 200), (900, 200), (60, 60, 60), 1)

    # Menu items
    items = [
        ("1", "New calibration", (100, 255, 200)),
        ("2", "Load saved profile", (100, 255, 200)),
        ("3", "Recalibrate existing profile", (100, 255, 200)),
    ]

    y = 280
    for key, label, color in items:
        cv2.putText(frame, f"[{key}]", (400, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, label, (470, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
        y += 65

    cv2.putText(frame, "[ESC]  Exit", (400, 530),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)

    # Footer
    cv2.putText(frame, "v2.0 | Index Tap: Left | Two Fingers: Right", (370, 680),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

    return frame


# ============================================================
# PROFILE SELECTOR
# ============================================================
def render_profile_selector_frame(profiles, selected_index):
    frame = np.zeros((780, 1280, 3), dtype=np.uint8)

    # Gradient background
    for row in range(780):
        t = row / 780.0
        frame[row, :] = (int(15 + 20 * t), int(10 + 12 * t), int(18 + 25 * t))

    cv2.putText(frame, "Select Profile", (470, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 200), 3)

    if not profiles:
        cv2.putText(frame, "No profiles found", (430, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 100, 255), 2)
        cv2.putText(frame, "[ESC]  Back", (470, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
        return frame

    y = 220
    for i, profile in enumerate(profiles):
        if i == selected_index:
            # Highlighted selection bar
            cv2.rectangle(frame, (390, y - 30), (890, y + 10), (40, 60, 40), -1)
            color = (100, 255, 200)
            prefix = ">"
        else:
            color = (200, 200, 200)
            prefix = " "
        cv2.putText(frame, f"{prefix} {profile}", (400, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        y += 60

    cv2.putText(frame, "[W/S or UP/DOWN]  Navigate", (330, 660),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 2)
    cv2.putText(frame, "[ENTER]  Load", (330, 700),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 2)
    cv2.putText(frame, "[ESC]  Back", (620, 700),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 2)

    return frame


# ============================================================
# CALIBRATION (with progress bar and stability ring)
# ============================================================
def render_calibration_frame(
    screen_w,
    screen_h,
    target,
    current,
    total,
    preview=None,
    status="Look at the red dot",
    stability=None,      # float 0.0-1.0 or None
):
    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    # Subtle dark gradient
    for row in range(0, screen_h, 4):
        t = row / max(screen_h, 1)
        val = int(8 + 12 * t)
        frame[row:row+4, :] = (val, val - 2 if val > 5 else 0, val + 3)

    x = int(target[0] * screen_w)
    y = int(target[1] * screen_h)

    # Outer pulsing ring (animated via current index)
    pulse = 38 + int(10 * math.sin(current * 0.8))
    cv2.circle(frame, (x, y), pulse, (0, 0, 180), 2)

    # Stability ring — green if stable, amber if not
    if stability is not None:
        ring_color = (0, 255, 100) if stability > 0.7 else (0, 180, 255)
        ring_radius = 28
        # Draw partial ring based on stability
        angle_end = int(360 * stability)
        cv2.ellipse(frame, (x, y), (ring_radius, ring_radius),
                    0, 0, angle_end, ring_color, 3)
    else:
        cv2.circle(frame, (x, y), 28, (0, 0, 200), 2)

    # Center dot
    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

    # Crosshair
    cv2.line(frame, (x - 18, y), (x + 18, y), (0, 0, 200), 1)
    cv2.line(frame, (x, y - 18), (x, y + 18), (0, 0, 200), 1)

    # Keep the HUD opposite the target. Fixed overlays used to cover top-row
    # targets and made the bottom-right point disappear behind the preview.
    hud_at_bottom = target[1] < 0.5
    bar_y = screen_h - 145 if hud_at_bottom else 30
    text_y = bar_y + 45

    # Progress bar
    bar_w = screen_w - 80
    bar_h = 14
    bar_x = 40
    progress = (current + 1) / max(total, 1)
    fill_w = int(bar_w * progress)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (40, 40, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                  (80, 220, 160), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), 1)

    # Text overlay
    cv2.putText(frame, f"Calibration {current + 1}/{total}", (30, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    cv2.putText(frame, status, (30, text_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2)
    cv2.putText(frame, "[ESC] Exit", (30, text_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1)

    # Camera preview
    if preview is not None:
        preview_copy = preview.copy()
        cv2.putText(preview_copy, "Camera", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        preview_x = 20 if target[0] >= 0.5 else screen_w - 380
        preview_y = 170 if target[1] >= 0.5 else screen_h - 260
        _paste_preview(frame, preview_copy, preview_x, preview_y)

    return frame


# ============================================================
# TRACKING OVERLAY (improved HUD)
# ============================================================
def draw_tracking_overlay(frame, gaze_x, gaze_y, screen_x=None, screen_y=None, clutch_active=True):
    h, w = frame.shape[:2]

    # Semi-transparent status bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if clutch_active:
        cv2.putText(frame, "EYE CONTROL: ON", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 120), 2)
    else:
        cv2.putText(frame, "EYE CONTROL: OFF (RAISE INDEX FINGER)", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 160, 255), 2)

    cv2.putText(frame, f"Gaze: ({gaze_x:.3f}, {gaze_y:.3f})", (450, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if screen_x is not None and screen_y is not None and clutch_active:
        cv2.putText(frame, f"Cursor: ({int(screen_x)}, {int(screen_y)})", (650, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


# ============================================================
# ON-SCREEN TEXT INPUT
# ============================================================
def onscreen_text_input(window_name, prompt="Enter name:", default=""):
    """Capture text input inside the CV window using key events.
    Returns the entered string, or None if ESC was pressed.
    """
    text = default
    cursor_blink = 0

    while True:
        frame = np.zeros((300, 700, 3), dtype=np.uint8)

        # Background
        for row in range(300):
            t = row / 300.0
            frame[row, :] = (int(20 + 15 * t), int(15 + 10 * t), int(25 + 20 * t))

        cv2.putText(frame, prompt, (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 200), 2)

        # Text field background
        cv2.rectangle(frame, (38, 90), (662, 140), (50, 50, 50), -1)
        cv2.rectangle(frame, (38, 90), (662, 140), (100, 255, 200), 1)

        # Cursor blink
        cursor_blink += 1
        cursor_char = "|" if (cursor_blink // 15) % 2 == 0 else ""

        cv2.putText(frame, text + cursor_char, (50, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, "[ENTER] Confirm    [ESC] Cancel    [BACKSPACE] Delete",
                    (40, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            return None
        elif key == 13:  # ENTER
            return text.strip() if text.strip() else default
        elif key == 8:  # BACKSPACE
            text = text[:-1]
        elif 32 <= key <= 126:  # printable ASCII
            if len(text) < 30:
                text += chr(key)

    return text


# ============================================================
# VALIDATION RESULTS SCREEN
# ============================================================
def render_validation_results(screen_w, screen_h, avg_error, max_error, quality):
    """Render a nicer validation results screen."""
    result = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    # Gradient
    for row in range(screen_h):
        t = row / max(screen_h, 1)
        result[row, :] = (int(10 + 15 * t), int(8 + 10 * t), int(15 + 20 * t))

    # Title
    cv2.putText(result, "Validation Results", (screen_w // 2 - 220, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 200), 3)

    # Quality badge
    quality_colors = {
        "EXCELLENT": (0, 255, 100),
        "GOOD": (0, 220, 255),
        "AVERAGE": (0, 180, 255),
        "POOR": (80, 80, 255),
    }
    q_color = quality_colors.get(quality, (255, 255, 255))

    cv2.putText(result, quality, (screen_w // 2 - 100, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, q_color, 4)

    # Stats
    cv2.putText(result, f"Average Error: {avg_error:.1f} px",
                (screen_w // 2 - 200, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)

    cv2.putText(result, f"Max Error: {max_error:.1f} px",
                (screen_w // 2 - 160, 490),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)

    cv2.putText(result, "Press any key to continue",
                (screen_w // 2 - 200, 600),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)

    return result

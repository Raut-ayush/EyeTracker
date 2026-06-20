# ============================================================
# main.py — Main application loop and logic
# ============================================================

import cv2
import time
import traceback
import pyautogui
import numpy as np

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CALIBRATION_SAMPLES,
    CALIBRATION_TIMEOUT,
    HEAD_STABILITY_YAW,
    HEAD_STABILITY_PITCH,
    DEAD_ZONE,
    SCREEN_MARGIN,
    NEUTRAL_POSE_FRAMES,
)

from eye_tracker import EyeTracker
from hand_tracker import HandTracker
from calibration import GazeCalibration
from filters import KalmanFilter2D, EdgeDamper, AdaptiveEMA2D
from profile_manager import ProfileManager
from ui import (
    render_main_menu_frame,
    render_profile_selector_frame,
    render_calibration_frame,
    draw_tracking_overlay,
    onscreen_text_input,
    render_validation_results,
)

WINDOW_NAME = "EyeTracker"
pyautogui.FAILSAFE = True


def log(message):
    print(f"[LOG] {message}")


def error_log(e):
    print("\n========== ERROR ==========")
    print(str(e))
    traceback.print_exc()
    print("===========================\n")


def setup_window(fullscreen=False):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if fullscreen:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )
    else:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL
        )
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    log("Window initialized")


def capture_neutral_pose(cap, tracker, screen_w, screen_h):
    """Ask user to look at screen center and capture neutral head pose.
    
    Returns (yaw, pitch) average or (0.0, 0.0) if capture fails.
    """
    log("Capturing neutral head pose...")
    yaw_samples = []
    pitch_samples = []
    
    start = time.time()
    while len(yaw_samples) < NEUTRAL_POSE_FRAMES and (time.time() - start) < 5.0:
        ret, cam = cap.read()
        if not ret:
            continue
        
        cam = cv2.flip(cam, 1)
        features = tracker.get_gaze_features(cam)
        
        # Draw instruction screen
        frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        for row in range(0, screen_h, 4):
            t = row / max(screen_h, 1)
            val = int(8 + 12 * t)
            frame[row:row+4, :] = (val, val - 2 if val > 5 else 0, val + 3)
        
        # Center target
        cx, cy = screen_w // 2, screen_h // 2
        cv2.circle(frame, (cx, cy), 30, (0, 180, 80), 2)
        cv2.circle(frame, (cx, cy), 10, (0, 255, 120), -1)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
        
        cv2.putText(frame, "Look at the center dot", (cx - 200, cy - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        cv2.putText(frame, "Keep your head in a natural position", (cx - 280, cy - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        
        progress = len(yaw_samples) / NEUTRAL_POSE_FRAMES
        bar_w = 300
        cv2.rectangle(frame, (cx - bar_w//2, cy + 60), (cx + bar_w//2, cy + 74),
                      (40, 40, 40), -1)
        cv2.rectangle(frame, (cx - bar_w//2, cy + 60), (cx - bar_w//2 + int(bar_w * progress), cy + 74),
                      (80, 220, 160), -1)
        
        # Camera preview
        if features is not None:
            tracker.draw_debug(cam, features)
            yaw_samples.append(float(features["yaw"]))
            pitch_samples.append(float(features["pitch"]))
        
        preview_h, preview_w = cam.shape[:2]
        scale = min(320 / preview_w, 200 / preview_h, 1.0)
        small = cv2.resize(cam, (int(preview_w * scale), int(preview_h * scale)))
        sh, sw = small.shape[:2]
        frame[screen_h - sh - 20:screen_h - 20, screen_w - sw - 20:screen_w - 20] = small
        
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return 0.0, 0.0
    
    if not yaw_samples:
        log("WARNING: No face detected during neutral pose capture")
        return 0.0, 0.0
    
    avg_yaw = sum(yaw_samples) / len(yaw_samples)
    avg_pitch = sum(pitch_samples) / len(pitch_samples)
    log(f"Neutral pose: yaw={avg_yaw:.4f}, pitch={avg_pitch:.4f} ({len(yaw_samples)} samples)")
    return avg_yaw, avg_pitch


def menu_loop():
    log("Opening menu")
    while True:
        frame = render_main_menu_frame()
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            return "new"
        elif key == ord("2"):
            return "load"
        elif key == ord("3"):
            return "recalibrate"
        elif key == 27:
            return None


def profile_selector(profile_manager):
    profiles = profile_manager.list_profiles()

    if not profiles:
        return None

    selected = 0
    while True:
        frame = render_profile_selector_frame(profiles, selected)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            return None
        elif key == 13:
            return profiles[selected]
        elif key in (82, ord("w")):
            selected = (selected - 1) % len(profiles)
        elif key in (84, ord("s")):
            selected = (selected + 1) % len(profiles)


def animate_target_transition(
    cap,
    tracker,
    screen_w,
    screen_h,
    start_target,
    end_target,
    current,
    total,
    steps=14
):
    for i in range(steps):
        a = i / max(steps - 1, 1)
        target = (
            start_target[0] + (end_target[0] - start_target[0]) * a,
            start_target[1] + (end_target[1] - start_target[1]) * a,
        )

        ret, cam = cap.read()
        if not ret:
            continue

        cam = cv2.flip(cam, 1)
        features = tracker.get_gaze_features(cam)

        preview = cam.copy()
        if features is not None:
            tracker.draw_debug(preview, features)

        frame = render_calibration_frame(
            screen_w,
            screen_h,
            target,
            current,
            total,
            preview=preview,
            status="Follow the moving dot"
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False

    return True


def _average_numeric_features(samples):
    if not samples:
        return None

    numeric_keys = []
    first = samples[0]

    for k, v in first.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            numeric_keys.append(k)

    if not numeric_keys:
        return None

    out = {}
    for k in numeric_keys:
        out[k] = float(sum(float(s[k]) for s in samples) / len(samples))
    return out


def capture_calibration_sample(
    cap,
    tracker,
    screen_w,
    screen_h,
    target,
    current,
    total,
    sample_count=CALIBRATION_SAMPLES,
    timeout_sec=CALIBRATION_TIMEOUT
):
    samples = []
    start = time.time()
    
    # Store reference eye height during the first central point
    is_center_point = (0.45 <= target[0] <= 0.55) and (0.45 <= target[1] <= 0.55)

    # Debug counters
    no_face_count = 0
    unstable_count = 0

    while len(samples) < sample_count and (time.time() - start) < timeout_sec:
        ret, cam = cap.read()
        if not ret:
            continue

        cam = cv2.flip(cam, 1)
        features = tracker.get_gaze_features(cam)

        preview = cam.copy()
        
        # Calculate stability metric for the ring
        stability = 0.0
        if features is None:
            no_face_count += 1
        elif features is not None:
            tracker.draw_debug(preview, features)
            
            # stability: 1.0 = perfectly still, 0.0 = moving head
            yaw_dev = abs(float(features.get("yaw", 0.0)))
            pitch_dev = abs(float(features.get("pitch", 0.0)))
            
            s_yaw = max(0.0, 1.0 - (yaw_dev / HEAD_STABILITY_YAW))
            s_pitch = max(0.0, 1.0 - (pitch_dev / HEAD_STABILITY_PITCH))
            stability = min(s_yaw, s_pitch)
            
            if stability > 0.05:
                samples.append(features)
            else:
                unstable_count += 1

        frame = render_calibration_frame(
            screen_w,
            screen_h,
            target,
            current,
            total,
            preview=preview,
            status=f"Hold still... {len(samples)}/{sample_count}",
            stability=stability
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None

    # Accept partial samples if we have at least 5
    if len(samples) < 5:
        log(f"Point {current+1} FAILED: {len(samples)} samples "
            f"(no_face={no_face_count}, unstable={unstable_count})")
        return None
    
    if len(samples) < sample_count:
        log(f"Point {current+1}: partial {len(samples)}/{sample_count} "
            f"(no_face={no_face_count}, unstable={unstable_count})")

    avg_features = _average_numeric_features(samples)

    if avg_features is None:
        return None
        
    # If this is a center point, update the tracker's reference eye height
    if is_center_point and "avg_eye_h" in avg_features:
        tracker.set_reference_eye_height(avg_features["avg_eye_h"])
        log(f"Set reference eye height: {avg_features['avg_eye_h']:.3f}")

    log(
        f"Avg gaze: ({avg_features['gaze_x']:.3f}, {avg_features['gaze_y']:.3f}) | "
        f"Comp: ({avg_features.get('comp_gx', 0):.3f}, {avg_features.get('comp_gy', 0):.3f}) | "
        f"Head: ({avg_features.get('yaw',0):.3f}, {avg_features.get('pitch',0):.3f})"
    )
    return avg_features


def calibration_flow(cap, tracker, screen_w, screen_h):
    # Reset reference eye height and neutral pose for fresh calibration
    tracker.set_reference_eye_height(None)
    tracker.set_neutral_pose(0.0, 0.0)
    
    # Step 1: Initialize calibration with zero neutral pose
    calibration = GazeCalibration()
    calibration.neutral_yaw = 0.0
    calibration.neutral_pitch = 0.0
    
    total = len(calibration.targets)

    prev_target = calibration.targets[0]

    for idx, target in enumerate(calibration.targets):
        if idx > 0:
            ok = animate_target_transition(
                cap,
                tracker,
                screen_w,
                screen_h,
                prev_target,
                target,
                idx,
                total
            )
            if not ok:
                log("Calibration cancelled")
                return None

        sample = capture_calibration_sample(
            cap,
            tracker,
            screen_w,
            screen_h,
            target,
            idx,
            total
        )

        if sample is None:
            log("Calibration cancelled or failed")
            return None

        if idx == 0:
            # Point 1 is the center target. Use its head pose as baseline!
            neutral_yaw = sample["yaw"]
            neutral_pitch = sample["pitch"]
            tracker.set_neutral_pose(neutral_yaw, neutral_pitch)
            calibration.neutral_yaw = neutral_yaw
            calibration.neutral_pitch = neutral_pitch
            log(f"Set neutral head pose baseline from center: yaw={neutral_yaw:.4f}, pitch={neutral_pitch:.4f}")

        calibration.add_sample(sample)
        log(f"Saved point {calibration.current_index}/{total}")
        prev_target = target

    calibration.fit()
    log("Calibration complete")
    return calibration


def save_profile_flow(profile_manager, calibration, default_name="default"):
    # Clear any residual keys
    cv2.waitKey(100)
    
    while True:
        name = onscreen_text_input(
            WINDOW_NAME,
            prompt="Enter profile name to save:",
            default=default_name
        )
        
        if name is None:  # ESC pressed
            return None
            
        name = name.strip()
        if not name:
            name = default_name

        existing = profile_manager.load_profile(name)
        if existing is not None:
            # Simple confirmation inline
            confirm = onscreen_text_input(
                WINDOW_NAME,
                prompt=f"Profile '{name}' exists. Overwrite? (y/n):",
                default="y"
            )
            if confirm is None or confirm.strip().lower() != "y":
                continue

        profile_manager.save_profile(name, calibration.to_dict())
        log(f"Saved profile: {name}")
        return name


def validation_mode(cap, tracker, calibration, screen_w, screen_h):
    log("Starting validation mode")

    targets = calibration.validation_targets()
    errors = []

    for idx, target in enumerate(targets):
        samples = []
        start = time.time()

        while time.time() - start < 2.0:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            features = tracker.get_gaze_features(frame)
            if features is not None:
                pred = calibration.predict(features)
                samples.append(pred)

            overlay = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            x = int(target[0] * screen_w)
            y = int(target[1] * screen_h)

            # Draw target
            cv2.circle(overlay, (x, y), 22, (0, 0, 255), -1)
            cv2.circle(overlay, (x, y), 8, (255, 255, 255), -1)
            
            cv2.putText(
                overlay,
                f"Validation {idx+1}/{len(targets)}",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            cv2.putText(
                overlay,
                "Look at the red target",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )
            cv2.putText(
                overlay,
                "ESC = Exit validation",
                (40, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 180, 180),
                2
            )

            cv2.imshow(WINDOW_NAME, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                log("Validation cancelled")
                return

        if len(samples) == 0:
            continue

        avg_x = sum(s[0] for s in samples) / len(samples)
        avg_y = sum(s[1] for s in samples) / len(samples)

        error = calibration.compute_error(
            (avg_x, avg_y),
            target,
            screen_w,
            screen_h
        )

        errors.append(error)
        log(f"[Validation {idx+1}] Error = {error:.1f}px")

    if len(errors) == 0:
        return

    avg_error = sum(errors) / len(errors)
    max_error = max(errors)
    quality = calibration.score_quality(avg_error)

    log("========== VALIDATION ==========")
    log(f"Average Error: {avg_error:.1f}px")
    log(f"Max Error: {max_error:.1f}px")
    log(f"Quality: {quality}")
    log("================================")

    result_frame = render_validation_results(
        screen_w, screen_h, avg_error, max_error, quality
    )
    cv2.imshow(WINDOW_NAME, result_frame)
    cv2.waitKey(0)


def tracking_loop(cap, tracker, hand_tracker, calibration, screen_w, screen_h):
    log("Tracking started")

    # Initialize new filter stack
    kf = KalmanFilter2D()
    adaptive_ema = AdaptiveEMA2D(alpha_low=0.03, alpha_high=0.35, speed_threshold=15.0)
    edge_damper = EdgeDamper(screen_w, screen_h)

    prev_x = screen_w / 2
    prev_y = screen_h / 2
    
    # Restore reference eye height if profile has it (via mean_ipd check or similar in future)
    # For now, it will self-calibrate slowly or run un-normalised if loaded from disk.

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        
        # 1. PROCESS GAZE
        features = tracker.get_gaze_features(frame)
        
        # 2. PROCESS HAND GESTURES
        hand_result = hand_tracker.process(frame)
        
        # Check for pending single taps (if double tap window expired)
        if hand_result is None:
            hand_result = hand_tracker.check_pending_tap()

        # Execute clicks
        if hand_result:
            if hand_result["gesture"] == "click":
                log("GESTURE: Single Click")
                pyautogui.click()
            elif hand_result["gesture"] == "doubleclick":
                log("GESTURE: Double Click")
                pyautogui.doubleClick()

        clutch_active = hand_tracker.is_index_finger_present()

        # Update cursor if gaze is valid (i.e., not blinking)
        if features is not None:
            tracker.draw_debug(frame, features)
            
            # Predict normalised coords
            norm_x, norm_y = calibration.predict(features)

            # Apply Kalman Filter in normalised coordinate space
            smooth_nx, smooth_ny = kf.update(norm_x, norm_y)

            # Convert to pixel space
            target_x = smooth_nx * screen_w
            target_y = smooth_ny * screen_h

            # Apply Adaptive EMA (fixation lock) in pixel space
            smooth_x, smooth_y = adaptive_ema.update(target_x, target_y)
            
            # Apply Edge Damping
            smooth_x, smooth_y = edge_damper.update(smooth_x, smooth_y)

            # Clamp to screen margins
            smooth_x = max(SCREEN_MARGIN, min(screen_w - SCREEN_MARGIN - 1, smooth_x))
            smooth_y = max(SCREEN_MARGIN, min(screen_h - SCREEN_MARGIN - 1, smooth_y))

            # Move cursor only if index finger is visible (clutch is active)
            if clutch_active:
                if abs(smooth_x - prev_x) > DEAD_ZONE or abs(smooth_y - prev_y) > DEAD_ZONE:
                    pyautogui.moveTo(int(smooth_x), int(smooth_y), duration=0)
                    prev_x, prev_y = smooth_x, smooth_y

            draw_tracking_overlay(
                frame,
                features["gaze_x"],
                features["gaze_y"],
                smooth_x,
                smooth_y,
                clutch_active=clutch_active
            )

        # Draw hand tracker debug
        hand_tracker.draw_debug(frame)

        cv2.putText(
            frame,
            "[F] Validation Mode",
            (30, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        cv2.putText(
            frame,
            "[ESC] Exit",
            (30, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('f'):
            validation_mode(
                cap,
                tracker,
                calibration,
                screen_w,
                screen_h
            )
            # Reset filters after validation to avoid sudden jumps
            kf.reset()
            adaptive_ema.reset()
            edge_damper.reset()

        elif key == 27:
            log("ESC pressed")
            break


def main():
    try:
        log("Starting EyeTracker v2")

        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        # Request high resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        if not cap.isOpened():
            log("Could not open webcam")
            return

        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        log(f"Webcam opened at {actual_w}x{actual_h}")

        screen_w, screen_h = pyautogui.size()
        log(f"Screen size: {screen_w}x{screen_h}")

        setup_window(fullscreen=False)

        tracker = EyeTracker()
        hand_tracker = HandTracker()
        profile_manager = ProfileManager()

        log("Trackers initialized")

        mode = menu_loop()
        if mode is None:
            cap.release()
            cv2.destroyAllWindows()
            return

        calibration = None

        if mode == "load":
            selected = profile_selector(profile_manager)
            if selected is None:
                cap.release()
                cv2.destroyAllWindows()
                return

            data = profile_manager.load_profile(selected)
            if data is None:
                log("Profile not found")
                cap.release()
                cv2.destroyAllWindows()
                return

            calibration = GazeCalibration.from_dict(data)
            tracker.set_neutral_pose(calibration.neutral_yaw, calibration.neutral_pitch)
            log(f"Loaded profile: {selected}")
            
            # Switch to UI for tracking
            setup_window(fullscreen=False)

        elif mode in ("new", "recalibrate"):
            
            default_name = "default"
            if mode == "recalibrate":
                selected = profile_selector(profile_manager)
                if selected is None:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                default_name = selected
                log(f"Recalibrating profile: {selected}")
            
            setup_window(fullscreen=True)

            calibration = calibration_flow(cap, tracker, screen_w, screen_h)
            if calibration is None:
                cap.release()
                cv2.destroyAllWindows()
                return

            saved_name = save_profile_flow(profile_manager, calibration, default_name=default_name)
            if not saved_name:
                log("Profile save cancelled")
            
            setup_window(fullscreen=False)

        else:
            cap.release()
            cv2.destroyAllWindows()
            return

        tracking_loop(cap, tracker, hand_tracker, calibration, screen_w, screen_h)

        cap.release()
        cv2.destroyAllWindows()
        log("Closed safely")

    except Exception as e:
        error_log(e)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
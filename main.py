# ============================================================
# main.py — Main application loop and logic
# ============================================================

import cv2
import time
import traceback
import pyautogui
import numpy as np
from collections import deque

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
    PYAUTOGUI_PAUSE,
    CALIBRATION_SETTLE_SECONDS,
    CALIBRATION_STABILITY_WINDOW,
    GAZE_STABILITY_X,
    GAZE_STABILITY_Y,
    CURSOR_ALPHA_LOW,
    CURSOR_ALPHA_HIGH,
    CURSOR_SPEED_THRESHOLD,
    CALIBRATION_MAX_RETRIES,
    CALIBRATION_RETRY_ERROR_RATIO,
    CALIBRATION_RETRY_MIN_IMPROVEMENT,
)

from eye_tracker import EyeTracker
from hand_tracker import HandTracker
from calibration import GazeCalibration
from filters import KalmanFilter2D, EdgeDamper, AdaptiveEMA2D, MedianFilter2D
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
pyautogui.PAUSE = PYAUTOGUI_PAUSE


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

    # Reject whole frames that are outliers in the signals most relevant to
    # gaze. This avoids mixing a blink/saccade into otherwise good averages.
    core_keys = ["gaze_x", "gaze_y", "yaw", "pitch"]
    core = np.array([[float(s[k]) for k in core_keys] for s in samples])
    median = np.median(core, axis=0)
    mad = np.median(np.abs(core - median), axis=0)
    robust_scale = np.maximum(1.4826 * mad, 1e-5)
    keep = np.all(np.abs(core - median) / robust_scale < 3.5, axis=1)
    filtered = [s for s, accepted in zip(samples, keep) if accepted]
    if len(filtered) < 5:
        filtered = samples

    out = {}
    for k in numeric_keys:
        out[k] = float(np.median([float(s[k]) for s in filtered]))
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
    timeout_sec=CALIBRATION_TIMEOUT,
    status_prefix="Hold still",
):
    samples = []
    start = time.time()
    
    # Store reference eye height during the first central point
    is_center_point = (0.45 <= target[0] <= 0.55) and (0.45 <= target[1] <= 0.55)

    # Debug counters
    no_face_count = 0
    unstable_count = 0
    recent = deque(maxlen=CALIBRATION_STABILITY_WINDOW)

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
            recent.append(features)

            settled = (time.time() - start) >= CALIBRATION_SETTLE_SECONDS
            if settled and len(recent) == CALIBRATION_STABILITY_WINDOW:
                yaw_std = np.std([s["yaw"] for s in recent])
                pitch_std = np.std([s["pitch"] for s in recent])
                gx_std = np.std([s["gaze_x"] for s in recent])
                gy_std = np.std([s["gaze_y"] for s in recent])
                ratios = (
                    yaw_std / HEAD_STABILITY_YAW,
                    pitch_std / HEAD_STABILITY_PITCH,
                    gx_std / GAZE_STABILITY_X,
                    gy_std / GAZE_STABILITY_Y,
                )
                stability = max(0.0, 1.0 - max(ratios))

            if settled and stability > 0.05:
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
            status=f"{status_prefix}... {len(samples)}/{sample_count}",
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
        # Persist this value so loaded profiles use identical normalization.
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
            calibration.reference_eye_height = sample.get("avg_eye_h")
            log(f"Set neutral head pose baseline from center: yaw={neutral_yaw:.4f}, pitch={neutral_pitch:.4f}")

        calibration.add_sample(sample)
        log(f"Saved point {calibration.current_index}/{total}")
        prev_target = target

    calibration.fit()

    # A point can be stable yet still be wrong (missed target, brief loss of
    # attention, unusual pose). Retry only strong post-fit outliers, and keep
    # the replacement only when it measurably improves the complete model.
    retry_threshold = (
        CALIBRATION_RETRY_ERROR_RATIO
        * float(np.hypot(screen_w, screen_h))
    )

    for retry_number in range(CALIBRATION_MAX_RETRIES):
        old_errors = calibration.training_errors(screen_w, screen_h)
        # The center establishes normalization and is deliberately not retried.
        worst_idx = 1 + int(np.argmax(old_errors[1:]))
        worst_error = float(old_errors[worst_idx])

        if worst_error <= retry_threshold:
            break

        target = calibration.targets[worst_idx]
        log(
            f"Calibration quality retry {retry_number + 1}/"
            f"{CALIBRATION_MAX_RETRIES}: point {worst_idx + 1} "
            f"residual={worst_error:.1f}px"
        )

        ok = animate_target_transition(
            cap,
            tracker,
            screen_w,
            screen_h,
            prev_target,
            target,
            worst_idx,
            total,
        )
        if not ok:
            log("Calibration retry cancelled; keeping original sample")
            break

        replacement = capture_calibration_sample(
            cap,
            tracker,
            screen_w,
            screen_h,
            target,
            worst_idx,
            total,
            status_prefix="Quality retry - hold still",
        )
        if replacement is None:
            log("Calibration retry failed; keeping original sample")
            break

        old_sample = calibration.samples[worst_idx].copy()
        old_score = float(np.mean(old_errors) + np.max(old_errors))
        calibration.replace_sample(worst_idx, replacement)
        calibration.fit()
        new_errors = calibration.training_errors(screen_w, screen_h)
        new_score = float(np.mean(new_errors) + np.max(new_errors))
        improvement = (old_score - new_score) / max(old_score, 1e-6)

        if improvement >= CALIBRATION_RETRY_MIN_IMPROVEMENT:
            log(
                f"Accepted retry for point {worst_idx + 1}: "
                f"fit improved {improvement * 100:.1f}%"
            )
            prev_target = target
        else:
            calibration.samples[worst_idx] = old_sample
            calibration.fit()
            log(
                f"Rejected retry for point {worst_idx + 1}: "
                f"fit improvement {improvement * 100:.1f}%"
            )
            break

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
            # Ignore the initial saccade and collect only the fixation period.
            if features is not None and time.time() - start >= 0.6:
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

        avg_x = float(np.median([s[0] for s in samples]))
        avg_y = float(np.median([s[1] for s in samples]))

        error = calibration.compute_error(
            (avg_x, avg_y),
            target,
            screen_w,
            screen_h
        )

        errors.append(error)
        err_x = (avg_x - target[0]) * screen_w
        err_y = (avg_y - target[1]) * screen_h
        log(f"[Validation {idx+1}] Error = {error:.1f}px "
            f"(dx={err_x:+.1f}, dy={err_y:+.1f})")

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
    median_filter = MedianFilter2D()
    kf = KalmanFilter2D()
    adaptive_ema = AdaptiveEMA2D(
        alpha_low=CURSOR_ALPHA_LOW,
        alpha_high=CURSOR_ALPHA_HIGH,
        speed_threshold=CURSOR_SPEED_THRESHOLD,
    )
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
        
        # Execute clicks
        if hand_result:
            if hand_result["gesture"] == "click":
                log("GESTURE: Single Click")
                pyautogui.click()
            elif hand_result["gesture"] == "rightclick":
                log("GESTURE: Right Click")
                pyautogui.rightClick()

        clutch_active = hand_tracker.is_index_finger_present()

        # Update cursor if gaze is valid (i.e., not blinking)
        if features is not None:
            tracker.draw_debug(frame, features)
            
            # Predict normalised coords
            norm_x, norm_y = calibration.predict(features)

            # Reject one- or two-frame involuntary gaze spikes before they can
            # add velocity to the Kalman state.
            norm_x, norm_y = median_filter.update(norm_x, norm_y)

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
            median_filter.reset()
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
            tracker.set_reference_eye_height(calibration.reference_eye_height)
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

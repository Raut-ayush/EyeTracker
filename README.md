# EyeTracker AI Mouse

EyeTracker AI Mouse is a webcam-based accessibility experiment that controls the
mouse pointer with gaze and uses simple hand gestures for clicking. It runs
locally with OpenCV, MediaPipe, NumPy, and PyAutoGUI; no dedicated eye-tracking
hardware is required.

> This is an experimental webcam tracker, not a medical or safety-critical
> assistive device. It works best with large interface targets and a stable
> camera setup.

## Features

- Real-time face, iris, and head-pose tracking with MediaPipe Face Mesh
- 17-point fullscreen gaze calibration
- Stable-sample collection, robust median averaging, and outlier rejection
- Automatic retry of unusually inaccurate calibration points
- Standardized ridge-regression gaze mapping
- Blink rejection and saved eye-height/head-pose normalization
- Median, Kalman, adaptive EMA, dead-zone, and edge smoothing
- Hand-controlled gaze clutch and mouse clicks
- Named calibration profiles with compatibility refitting for older profiles
- Six-point validation with total, horizontal, and vertical pixel error
- On-screen camera and tracking diagnostics

## Gesture controls

| Hand pose | Action |
| --- | --- |
| No extended index finger | Pause gaze-controlled cursor movement |
| Index finger extended | Enable gaze-controlled cursor movement |
| Index-only down/up tap | Left click immediately |
| Index and middle raised; ring and pinky folded | Right click once |

The two-finger pose freezes gaze movement while the right click is issued. Lower
the pose before raising it again; holding it up will not repeatedly click.

## Keyboard controls

| Screen | Key | Action |
| --- | --- | --- |
| Main menu | `1` | Create a new calibration |
| Main menu | `2` | Load a saved profile |
| Main menu | `3` | Recalibrate an existing profile |
| Profile selector | `W` / `S` or arrow keys | Change selection |
| Profile selector | `Enter` | Load selected profile |
| Tracking | `F` | Run validation mode |
| Any main screen | `Esc` | Cancel, go back, or exit |

## Requirements

- A webcam, preferably fixed above or below the display
- A reasonably well-lit face with both eyes visible
- Python with support for the MediaPipe version installed on your platform
- Windows, macOS, or Linux with permission to control the system mouse

Python packages are listed in `requirements.txt`:

- `opencv-python`
- `mediapipe`
- `numpy`
- `pyautogui`

## Installation

Clone or download the project, open a terminal in its directory, and create a
virtual environment. On Windows PowerShell:

```powershell
python -m venv eyetracker
.\eyetracker\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
python3 -m venv eyetracker
source eyetracker/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running

```powershell
python main.py
```

The application requests a 1280x720 camera stream and reports the actual camera
and screen resolution in the terminal.

### First-time workflow

1. Select **New calibration** from the main menu.
2. Sit in a comfortable position and keep the webcam and chair stationary.
3. Follow each target with your eyes. Natural small head movement is acceptable,
   but avoid changing posture or distance.
4. Hold each target until its sample counter finishes.
5. If a point is a strong fitted outlier, the application may show it again as a
   **Quality retry**. The replacement is kept only when the overall fit improves.
6. Save the calibration under a profile name.
7. Raise only your index finger to enable gaze movement.
8. Press `F` during tracking to measure validation accuracy.

For best results, keep the same camera position, screen resolution, lighting,
seating distance, and posture used during calibration. Recalibrate after any
substantial change.

## Validation

Validation displays six targets that are separate from the calibration points.
Each terminal result contains total error and signed axis errors:

```text
[Validation 1] Error = 190.4px (dx=-100.1, dy=-162.0)
```

- `Error` is the Euclidean distance between prediction and target.
- `dx` is horizontal bias: negative is left, positive is right.
- `dy` is vertical bias: negative is up, positive is down.

The final quality labels are based on average pixel error:

| Average error | Quality |
| --- | --- |
| Below 80 px | Excellent |
| Below 150 px | Good |
| Below 250 px | Average |
| 250 px or above | Poor |

Validation measures gaze-model accuracy before cursor smoothing. The visible
cursor may therefore feel steadier than the raw validation score suggests.

## How it works

```text
Webcam frame
    |
    +--> Face Mesh --> iris/head/blink features --> calibration model
    |                                             |
    |                                             v
    |                                  median + Kalman + adaptive EMA
    |                                             |
    +--> Hand landmarks --> clutch/click gesture  v
                                         system mouse via PyAutoGUI
```

The face frame is downscaled to 640x360 for inference. Iris position is
normalized inside each eye, blink frames are discarded, and head pose is used as
an additional calibration signal. The compact model deliberately uses fewer
features than calibration points to reduce memorization and improve behavior
between targets.

During tracking, a five-frame median filter rejects brief gaze spikes. A
constant-velocity Kalman filter and adaptive EMA then stabilize fixation while
allowing larger gaze shifts to move continuously. Extra damping is applied near
screen edges.

## Profiles

Profiles are JSON files stored in `profiles/`. They include calibration samples,
model coefficients, neutral head pose, eye-height reference, and feature
normalization values.

When an older profile schema is loaded, the application refits its saved samples
with the current model and prints a recalibration recommendation. A fresh profile
is still preferable after major tracking changes.

## Configuration

Common tuning options live in `config.py`:

| Setting | Purpose |
| --- | --- |
| `CAMERA_INDEX` | Select the webcam |
| `DEAD_ZONE` | Ignore very small cursor movements |
| `GAZE_MEDIAN_WINDOW` | Reject short involuntary gaze spikes |
| `CURSOR_ALPHA_LOW` | Fixation stability; lower is steadier and slower |
| `CURSOR_ALPHA_HIGH` | Responsiveness during large gaze shifts |
| `KALMAN_PROCESS_NOISE` | Motion responsiveness of the Kalman filter |
| `KALMAN_MEASUREMENT_NOISE` | How strongly sensor noise is rejected |
| `CALIBRATION_SAMPLES` | Accepted frames collected per target |
| `CALIBRATION_MAX_RETRIES` | Maximum post-fit outlier retries |
| `TAP_DOWN_THRESHOLD` | Required downward index movement |
| `TAP_UP_THRESHOLD` | Required return movement to left-click |
| `TAP_COOLDOWN_MS` | Prevent repeated click triggers |
| `HAND_DETECT_EVERY_N_FRAMES` | Hand-tracking CPU/performance tradeoff |
| `EYE_MODE` | Use both, left, or right eye signal |

Change one group of settings at a time and keep validation logs so results can be
compared objectively.

## Project structure

```text
EyeTracker/
|-- main.py              Application flow, calibration, validation, tracking
|-- eye_tracker.py       Face Mesh eye/head feature extraction
|-- hand_tracker.py      Gaze clutch and click gesture state machine
|-- calibration.py       Calibration model, fitting, prediction, serialization
|-- filters.py           Median, Kalman, EMA, and edge filters
|-- config.py            Camera, calibration, filter, and gesture settings
|-- profile_manager.py   Save/load/list profile JSON files
|-- ui.py                OpenCV menus, targets, overlays, and result screens
|-- utils.py             General helper functions
|-- requirements.txt     Python dependencies
|-- profiles/            Saved user calibration profiles
`-- README.md
```

## Current limitations

- A normal webcam cannot match the precision of infrared eye-tracking hardware.
- Small buttons remain difficult; large targets are recommended.
- Accuracy can change with lighting, glasses glare, camera movement, posture,
  seating distance, facial angle, and screen resolution.
- Head direction can still influence the cursor more than desired.
- Vertical gaze generally has less reliable landmark range than horizontal gaze.
- Hand gestures require the relevant fingertips and joints to remain visible.
- A single-camera system may briefly lose tracking during blinks or occlusion.

## Troubleshooting

### Webcam does not open

- Close applications already using the camera.
- Check operating-system camera permissions.
- Try another value for `CAMERA_INDEX` in `config.py`.

### Cursor does not move

- Raise and straighten the index finger so its tip is above its base joint.
- Keep the hand visible inside the camera frame.
- Confirm that the operating system permits Python to control the mouse.
- Watch the overlay for `EYE CONTROL: ON`.

### Right click does not trigger

- Raise index and middle fingers together.
- Keep ring and pinky folded.
- Lower the two-finger pose before attempting another right click.

### Calibration is inconsistent

- Improve front lighting and reduce reflections on glasses.
- Keep the camera and seating position fixed.
- Look at the dot rather than the camera preview.
- Recalibrate instead of loading a profile created under different conditions.
- Review validation `dx` and `dy` values for systematic directional bias.

### Tracking is too shaky or too slow

- For more stability, lower `CURSOR_ALPHA_LOW` or increase
  `KALMAN_MEASUREMENT_NOISE` slightly.
- For faster response, raise `CURSOR_ALPHA_HIGH` or
  `KALMAN_PROCESS_NOISE` slightly.
- Avoid changing several filter values at once.

## Privacy

Camera frames are processed locally by the application. The project does not
contain code that uploads frames or calibration profiles to an external service.

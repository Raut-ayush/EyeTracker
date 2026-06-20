# EyeTracker AI Mouse

## Overview

EyeTracker AI Mouse is a webcam-based eye tracking system that allows users to control the mouse cursor using eye movement.

The system uses:
- OpenCV
- MediaPipe FaceMesh
- Iris landmark tracking
- Polynomial calibration mapping
- Moving average smoothing

---

# Features

- Real-time iris tracking
- Relative gaze estimation
- Fullscreen 9-point calibration
- Multi-sample averaging
- Cursor smoothing
- Dead-zone stabilization
- User profile save/load
- Modular architecture

---

# Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

---

# Project Workflow

```text
Webcam
   ↓
FaceMesh Detection
   ↓
Iris Landmark Extraction
   ↓
Relative Gaze Estimation
   ↓
Calibration Mapping
   ↓
Smoothing Filters
   ↓
Mouse Cursor Movement
```

```
EyeTracker/
│
├── main.py
├── eye_tracker.py
├── calibration.py
├── filters.py
├── profile_manager.py
├── ui.py
├── hand_tracker.py
├── utils.py
├── requirements.txt
│
├── profiles/
│
└── README.md
```
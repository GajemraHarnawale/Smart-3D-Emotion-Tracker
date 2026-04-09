
# Smart 3D Emotion Tracker

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Real-Time RGB-D Pipeline for Integrated Pose Estimation, Emotion Recognition, and Behavioral Tracking Using YOLOv8 and Microsoft Kinect**

> Gajemra Harnawale  
> Department of Electrical Engineering, Veermata Jijabai Technological Institute (VJTI), Mumbai, India

---

## Overview

This repository contains the complete open-source implementation of the Smart 3D Emotion Tracker system — a real-time multimodal behavioral analysis pipeline that integrates:

- **YOLOv8n-pose** — Human keypoint detection (17 COCO keypoints)
- **FER library (VGG-13)** — Facial emotion recognition (7 classes)
- **DeepSORT** — Persistent multi-object identity tracking
- **Microsoft Kinect Xbox 360 (OpenNI2)** — RGB-D depth sensing
- **3-Keypoint Geometric Gaze Estimator** — No training data required
- **Dual-format Behavioral Logger** — CSV + JSON per-track journals
- **Tkinter Annotation Interface** — Interactive operator notes

The system runs at **27 FPS** on a commodity laptop (Intel Core i7-10th gen, 16 GB RAM, NVIDIA MX350).

---

## System Architecture

```
Kinect Xbox 360 (OpenNI2)
        │
        ├── Colour Stream (640×480 @ 30 FPS)
        └── Depth Stream  (640×480 @ 30 FPS)
                │
                ▼
        YOLOv8n-pose ──► 17 Keypoints + BBox
                │
                ├──► FER Emotion Recognition ──► Top emotion label
                ├──► Depth Centroid Sampling ──► Depth in mm
                ├──► 3-Keypoint Gaze Estimator ──► Left / Centre / Right
                └──► DeepSORT Tracker ──► Persistent Track ID
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
             CSV Log File         JSON Emotion Journal
             (per-frame)          (per-track)
                    │
                    ▼
         3× OpenCV Display Windows + Matplotlib 3D Plot
```

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Depth Sensor | Microsoft Kinect Xbox 360 (Model 1473) |
| CPU | Intel Core i7 (8th gen or above recommended) |
| RAM | 16 GB minimum |
| GPU | NVIDIA GPU with CUDA support (MX350 or above) |
| OS | Ubuntu 22.04 (tested) / Windows 10+ |
| USB | USB 2.0 port for Kinect |

> **Cost:** Microsoft Kinect Xbox 360 is available on the used market for $50–80 USD, making this system deployable under $150 total hardware cost.

---

## Software Requirements

- Python 3.9
- Kinect Xbox 360 drivers (libfreenect or OpenNI2)
- CUDA-compatible GPU drivers

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/GajemraHarnawale/Smart-3D-Emotion-Tracker.git
cd Smart-3D-Emotion-Tracker
```

### Step 2 — Create a virtual environment (recommended)

```bash
python3.9 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install OpenNI2 Kinect drivers

Follow the OpenNI2 installation guide:
```
https://github.com/OpenNI/OpenNI2
```

For Ubuntu:
```bash
sudo apt-get install libopenni2-dev
```

### Step 5 — Connect your Kinect

Plug in the Microsoft Kinect Xbox 360 via USB 2.0 and verify it is detected:

```bash
lsusb | grep Microsoft
```

---

## Running the System

```bash
cd python_scripts
python main.py
```

### Runtime Controls

| Key | Action |
|-----|--------|
| `q` | Quit and save all logs |
| `a` | Open annotation dialog — attach a note to any Track ID |

### Output Windows

Three OpenCV windows open simultaneously:
- **Blended RGB+Depth** — 60% colour / 40% depth overlay with full annotations
- **Depth View** — False-colour JET depth map (blue=near, red=far)
- **RGB View** — Pure colour frame with skeleton and behavioral labels

A live **Matplotlib 3D scatter plot** shows subject centroid trajectories in real time.

---

## Output Files

All outputs are saved automatically to the `logs/` directory:

| File | Format | Contents |
|------|--------|----------|
| `emotion_pose_log_<timestamp>.csv` | CSV | Per-frame: Timestamp, Track ID, X, Y, Depth(mm), Emotion, Gaze, Note |
| `emotion_journal_<timestamp>.json` | JSON | Per-track emotion history with timestamps |

Sample log files from the single-subject validation protocol are included in `logs/`.

---

## Module Structure

```
python_scripts/
├── main.py                  # Main pipeline — run this
├── detector.py              # YOLOv8n-pose wrapper
├── emotion_recognition.py   # FER emotion classifier wrapper
├── gaze_estimation.py       # 3-keypoint geometric gaze estimator
├── tracker.py               # DeepSORT tracker wrapper
├── logger.py                # CSV + JSON dual-format logger
└── openni_setup.py          # Kinect / OpenNI2 initialisation

models/
└── yolov8n-pose.pt          # YOLOv8n-pose pretrained weights (COCO-2017)

logs/
├── emotion_pose_log.csv                      # Sample log (short)
└── emotion_pose_log_20250702_125207.csv      # Sample log (validation session)

Output/
├── 3d_trajectories.jpg          # 3D centroid trajectory plot
├── depth_skeleton.jpg           # Kinect depth + YOLOv8 skeleton overlay
├── emotion_distribution.jpg     # Emotion histogram (90-second segment)
├── integrated_view.jpg          # Blended RGB-depth view with annotations
└── realtime_processing.jpg      # RGB view with full behavioral overlay

data/
└── kinect_samples/              # Placeholder — see data/kinect_samples/README.md
```

---

## Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Gaze threshold | ±10 px | `gaze_estimation.py` |
| DeepSORT max_age | 30 frames (~1.1 s) | `tracker.py` |
| Depth sampling | Centroid (single point) | `main.py: get_depth_at()` |
| FPS (achieved) | 27.1 FPS | Measured over 90,000 frames |
| Operating depth range | 1.2–3.8 m | Kinect hardware limit |
| RGB-D resolution | 640×480 @ 30 FPS | OpenNI2 stream config |
| Emotion classes | 7 (angry, disgust, fear, happy, neutral, sad, surprise) | FER library |

---

## Validated Performance (Single-Subject Protocol)

| Metric | Value | 95% CI |
|--------|-------|--------|
| System throughput | 27.1 FPS | [26.7, 27.5] |
| Identity consistency (unoccluded) | 100% | — |
| Occlusion re-ID recovery (≤5 frames) | 94.2% | [91.3, 97.1] |
| Gaze accuracy vs. calibration markers | 89.3% | [86.1, 92.5] |
| YOLOv8n-pose AP@0.5 (COCO-2017) | 91.6% | [91.1, 92.1] |
| FER accuracy (AffectNet val) | 71.8% | [70.9, 72.7] |

> All protocol results derive from a controlled single-subject validation (50 min, 90,000 frames).  
> No multi-participant human subjects study was conducted.

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FER Library](https://github.com/justinshenk/fer)
- [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime)
- [OpenCV](https://opencv.org/)
- [OpenNI2](https://github.com/OpenNI/OpenNI2)
- Department of Electrical Engineering, VJTI Mumbai

---

## Contact

**Gajemra Harnawale** — gcharnawale_m23@et.vjti.ac.in  
Veermata Jijabai Technological Institute, Mumbai, India

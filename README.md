# Smart-3D-Emotion-Tracker
Real-time 3D emotion-aware surveillance system using YOLOv8, Kinect RGB-D sensing, and deep learning-based FER.
# Smart 3D Emotion Tracker: Real-Time Emotion-Aware Surveillance with YOLOv8 and Kinect

This repository contains source code, sample data, and documentation for the research paper:

**"Smart 3D Emotion Tracker: Real-Time Emotion-Aware Surveillance with YOLOv8 and Kinect"**

## Features
- YOLOv8-based real-time person detection
- Kinect RGB-D integration for 3D pose estimation
- Facial Emotion Recognition (FER) module
- DeepSORT for multi-person tracking
- GUI with emotion logging and annotation

## Data
- Kinect RGB-D sample frames included in `/data/kinect_samples/`
- FER-2013 dataset (public): https://www.kaggle.com/datasets/msambare/fer2013
- OpenNI2 Kinect drivers: https://github.com/OpenNI/OpenNI2

## Requirements
- Python 3.10+
- Ultralytics YOLOv8
- OpenNI2
- PyTorch
- OpenCV
- Tkinter (for GUI)

Install dependencies:
```bash
pip install -r requirements.txt

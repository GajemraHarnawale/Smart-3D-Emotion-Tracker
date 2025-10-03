# Smart-3D-Emotion-Tracker

**Kinect Xbox 360 Model 1473**

Real-time 3D emotion-aware surveillance system using YOLOv8, Kinect RGB-D sensing, and deep learning-based Facial Emotion Recognition (FER).

This repository contains source code, sample data, and documentation for the research paper:

**"Smart 3D Emotion Tracker: Real-Time Emotion-Aware Surveillance with YOLOv8 and Kinect"**

---

## Features

- YOLOv8-based real-time person detection  
- Kinect RGB-D integration for 3D pose estimation  
- Facial Emotion Recognition (FER) module  
- DeepSORT for multi-person tracking  
- GUI with real-time emotion logging and annotation  
- 3D visualization of tracked poses and emotions  

---

## Data

- Kinect RGB-D sample frames included in `/data/kinect_samples/`  
- FER-2013 dataset (public): https://www.kaggle.com/datasets/msambare/fer2013  
- OpenNI2 Kinect drivers: https://github.com/OpenNI/OpenNI2  

> Note: Ensure your Kinect Xbox 360 Model 1473 is connected and drivers are installed before running the system.

---

## Requirements

- Python 3.10+  
- Ultralytics YOLOv8  
- OpenNI2  
- PyTorch  
- OpenCV  
- Tkinter (for GUI)  
- deep_sort_realtime  
- fer  

Install dependencies:

```bash
pip install -r requirements.txt

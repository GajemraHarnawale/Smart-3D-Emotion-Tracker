Final_Mtech_project_second_year.py

import cv2
import numpy as np
import torch
import time
import json
import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
from fer import FER
from deep_sort_realtime.deepsort_tracker import DeepSort
from openni import openni2
from openni import _openni2 as c_api
import tkinter as tk
from tkinter.simpledialog import askstring

# === Init OpenNI2 ===
openni2.initialize()
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()
depth_stream.start()
color_stream.start()
dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

# === Load Models ===
pose_model = YOLO("yolov8n-pose.pt")  # Skeleton Pose
emotion_detector = FER()
tracker = DeepSort(max_age=30)

# === GUI + Logging Setup ===
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
csv_path = f"logs/emotion_pose_log_{timestamp}.csv"
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "ID", "X", "Y", "Depth", "Emotion", "Gaze", "Note"])

emotion_journal = {}
annotations = {}

# === Helper Functions ===
def get_depth_at(depth_frame, x, y):
    if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
        return depth_frame[y, x]
    return 0

def estimate_gaze(landmarks):
    if len(landmarks) < 3:
        return "Unknown"
    nose = landmarks[0]
    left_eye = landmarks[1]
    right_eye = landmarks[2]
    gaze_x = (left_eye[0] + right_eye[0]) / 2 - nose[0]
    if gaze_x > 10:
        return "Left"
    elif gaze_x < -10:
        return "Right"
    else:
        return "Center"

def draw_skeleton(img, keypoints):
    skeleton_pairs = [
        (5, 7), (7, 9),    # Left arm
        (6, 8), (8, 10),   # Right arm
        (5, 6), (5, 11),   # Shoulders
        (6, 12), (11, 13), # Torso and legs
        (13, 15), (12, 14), (14, 16),
        (11, 12)           # Pelvis
    ]
    for pair in skeleton_pairs:
        i, j = pair
        if i < len(keypoints) and j < len(keypoints):
            xi, yi = int(keypoints[i][0]), int(keypoints[i][1])
            xj, yj = int(keypoints[j][0]), int(keypoints[j][1])
            cv2.line(img, (xi, yi), (xj, yj), (255, 0, 0), 2)
            cv2.circle(img, (xi, yi), 3, (0, 255, 255), -1)
            cv2.circle(img, (xj, yj), 3, (0, 255, 255), -1)
    return img

# === Main Loop ===
while True:
    # RGB + Depth Frames
    color_frame = color_stream.read_frame()
    color_data = color_frame.get_buffer_as_uint8()
    color_img = np.frombuffer(color_data, dtype=np.uint8).reshape((480, 640, 3))
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    depth_frame = depth_stream.read_frame()
    depth_data = depth_frame.get_buffer_as_uint16()
    depth_img = np.frombuffer(depth_data, dtype=np.uint16).reshape((480, 640))

    depth_8bit = cv2.convertScaleAbs(depth_img, alpha=0.03)
    depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

    # === YOLOv8 Pose Estimation ===
    results = pose_model(color_img)[0]
    detections = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))

    # === DeepSORT Tracking ===
    tracks = tracker.update_tracks(detections, frame=color_img)

    for i, track in enumerate(tracks):
        if not track.is_confirmed():
            continue

        track_id = str(track.track_id)
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # === Skeleton (keypoints) ===
        keypoints = results.keypoints.xy[i].cpu().numpy() if len(results.keypoints.xy) > i else []
        if len(keypoints) > 0:
            color_img = draw_skeleton(color_img, keypoints)

        # === Depth Estimation ===
        depth = get_depth_at(depth_img, cx, cy)

        # === Emotion Detection ===
        face_crop = color_img[y1:y2, x1:x2]
        emotion = "Unknown"
        if face_crop.size != 0:
            try:
                emotion_data = emotion_detector.top_emotion(face_crop)
                if emotion_data:
                    emotion = emotion_data[0]
            except:
                pass

        # === Gaze Estimation ===
        gaze_dir = estimate_gaze(keypoints[:3]) if len(keypoints) >= 3 else "Unknown"

        # === Log and Journal ===
        if track_id not in emotion_journal:
            emotion_journal[track_id] = []
        emotion_journal[track_id].append({
            "time": datetime.now().isoformat(),
            "emotion": emotion,
            "gaze": gaze_dir,
            "depth": int(depth),
            "center": [int(cx), int(cy)]
        })

        note = annotations.get(track_id, "")
        label = f"ID:{track_id} {emotion} Gaze:{gaze_dir} Note:{note}"
        cv2.putText(color_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # === 3D Plot ===
        if depth > 0:
            ax.scatter(cx, cy, depth, label=f"ID {track_id}", s=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Depth')
            ax.set_title('3D Pose & Emotion Trajectories')
            plt.draw()
            plt.pause(0.001)

        # === CSV Log ===
        csv_writer.writerow([datetime.now().isoformat(), track_id, cx, cy, depth, emotion, gaze_dir, note])

    # === Display Views ===
    if depth_colored.shape != color_img.shape:
        depth_colored = cv2.resize(depth_colored, (color_img.shape[1], color_img.shape[0]))
    blended = cv2.addWeighted(color_img, 0.6, depth_colored, 0.4, 0)

    cv2.imshow("Blended RGB + Depth", blended)
    cv2.imshow("Depth View", depth_colored)
    cv2.imshow("RGB View", color_img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("a"):
        root = tk.Tk()
        root.withdraw()
        person_id = askstring("Input", "Enter Track ID:")
        if person_id:
            note = askstring("Input", f"Enter note for ID {person_id}:")
            if note:
                annotations[person_id] = note

# === Cleanup ===
csv_file.close()
openni2.unload()
cv2.destroyAllWindows()

with open(f"logs/emotion_journal_{timestamp}.json", "w") as f:
    json.dump(emotion_journal, f, indent=2)

print("✅ Tracking ended and data saved.")



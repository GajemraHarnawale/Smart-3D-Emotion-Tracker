import csv
import json
import os
from datetime import datetime

class Logger:
    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("logs", exist_ok=True)
        self.csv_path = f"logs/emotion_pose_log_{timestamp}.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Timestamp", "ID", "X", "Y", "Depth", "Emotion", "Gaze", "Note"])
        self.emotion_journal = {}

    def log(self, track_id, cx, cy, depth, emotion, gaze, note=""):
        timestamp = datetime.now().isoformat()
        self.csv_writer.writerow([timestamp, track_id, cx, cy, depth, emotion, gaze, note])
        if track_id not in self.emotion_journal:
            self.emotion_journal[track_id] = []
        self.emotion_journal[track_id].append({
            "time": timestamp,
            "emotion": emotion,
            "gaze": gaze,
            "depth": int(depth),
            "center": [int(cx), int(cy)]
        })

    def save_json(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/emotion_journal_{timestamp}.json", "w") as f:
            json.dump(self.emotion_journal, f, indent=2)
        self.csv_file.close()

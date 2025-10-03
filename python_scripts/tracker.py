from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age)

    def update_tracks(self, detections, frame):
        return self.tracker.update_tracks(detections, frame=frame)

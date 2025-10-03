def estimate_gaze(landmarks):
    if len(landmarks) < 3:
        return "Unknown"
    nose, left_eye, right_eye = landmarks[0], landmarks[1], landmarks[2]
    gaze_x = (left_eye[0] + right_eye[0]) / 2 - nose[0]
    if gaze_x > 10:
        return "Left"
    elif gaze_x < -10:
        return "Right"
    else:
        return "Center"

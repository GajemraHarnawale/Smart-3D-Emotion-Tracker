from fer import FER

class EmotionRecognizer:
    def __init__(self):
        self.detector = FER()

    def predict(self, face_crop):
        try:
            emotion_data = self.detector.top_emotion(face_crop)
            if emotion_data:
                return emotion_data[0]
        except:
            pass
        return "Unknown"

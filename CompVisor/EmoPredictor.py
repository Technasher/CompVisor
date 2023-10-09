from fer import FER


class EmoPredictor:
    def __init__(self):
        self.emotion_score = None
        self.dominant_emotion = None
        self.emo_predictor = FER()

    def predict_emo(self, img):
        self.dominant_emotion, self.emotion_score = self.emo_predictor.top_emotion(img)
        return dict(
            dominant_emotion=self.dominant_emotion,
            score=self.emotion_score
        )

import cv2

AGE_MODEL = 'CompVisor/AgePredictor/age_net.caffemodel'
AGE_PROTO = 'CompVisor/AgePredictor/deploy_age.prototxt'
AGE_INTERVALS = [(0, 2), (4, 6), (8, 12), (15, 20),
                 (25, 32), (38, 43), (48, 53), (60, 100)]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


class AgePredictor:
    def __init__(self):
        self.blob = None
        self._age_preds = None
        self._age_argmax = None
        self._age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

    def predict_age(self, img):
        self.blob = cv2.dnn.blobFromImage(
            image=img, scalefactor=1.0, size=(227, 227),
            mean=MODEL_MEAN_VALUES, swapRB=False
        )
        self._age_net.setInput(self.blob)
        self._age_preds = self._age_net.forward()
        self._age_argmax = self._age_preds[0].argmax()
        return dict(
            interval=AGE_INTERVALS[self._age_argmax],
            score=self._age_preds[0][self._age_argmax]
        )

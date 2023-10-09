import cv2

GENDER_MODEL = 'CompVisor/GenderPredictor/gender_net.caffemodel'
GENDER_PROTO = 'CompVisor/GenderPredictor/deploy_gender.prototxt'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


class GenderPredictor:
    def __init__(self):
        self.blob = None
        self._gender_preds = None
        self._gender_argmax = None
        self._gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

    def predict_gender(self, img):
        """Predict the gender of the faces showing in the image"""
        self.blob = cv2.dnn.blobFromImage(
            image=img, scalefactor=1.0, size=(227, 227),
            mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
        )
        # Predict Gender
        self._gender_net.setInput(self.blob)
        self._gender_preds = self._gender_net.forward()
        self._gender_argmax = self._gender_preds[0].argmax()
        return dict(
            gender=self._gender_argmax,
            score=self._gender_preds[0][self._gender_argmax]
        )

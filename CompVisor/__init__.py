from .EmoPredictor import EmoPredictor
from .AgePredictor import AgePredictor
from .GenderPredictor import GenderPredictor
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')


class CompVisor(EmoPredictor, AgePredictor, GenderPredictor):
    def __init__(self):
        EmoPredictor.__init__(self)
        AgePredictor.__init__(self)
        GenderPredictor.__init__(self)
        self.face_img = None
        self.face_gray = None
        self.face_x = None
        self.face_y = None

    def recognize(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        result = []
        for (x, y, w, h) in faces:
            self.face_x = x
            self.face_y = y
            self.face_img = img[y:y + h, x:x + w]
            self.face_gray = gray[y:y + h, x:x + w]
            result.append({
                'face_box': [x, y, w, h],
                'eye_boxes': self.detect_eyes(self.face_gray),
                'face_info': {
                    'emotion': self.predict_emo(self.face_img),
                    'age': self.predict_age(self.face_img),
                    'gender': self.predict_gender(self.face_img)
                }
            })
        return result

    def detect_eyes(self, face_img):
        eyes = eye_cascade.detectMultiScale(face_img, 3.5, 1)
        res = []
        for (x, y, w, h) in eyes:
            res.append([x + self.face_x, y + self.face_y, w, h])
        return res

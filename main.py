from CompVisor import CompVisor
from time import perf_counter
import cv2
from sys import argv

cap = cv2.VideoCapture(int(input('Choose camera: ') if len(argv) < 2 else argv[1]))
CoVi = CompVisor()


def percent(value):
    if value is None:
        return 0
    else:
        return value * 100


def draw_rect(x, y, w, h, color, bd=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, bd)


def draw_rect_with_text(x, y, w, h, color, label, bd=None):
    draw_rect(x, y, w, h, color, bd)
    infobox_h = 12
    lines = label.split('\n')
    cv2.rectangle(img, (x, y + h), (int(0.8 * w) + x, infobox_h * len(lines) + y + h + 3), color, -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + h + infobox_h * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, w / 550, (0, 0, 0))


def main():
    global ret, img
    while True:
        start = perf_counter()
        ret, img = cap.read()
        faces = CoVi.recognize(img)
        for face in faces:
            info = face['face_info']
            emotion = info['emotion']
            age = info['age']
            gender = info['gender']
            text = f'Emotion: {emotion["dominant_emotion"]} - {percent(emotion["score"]):.2f}%\n' \
                   f'Age: {age["interval"]} - {percent(age["score"]):.2f}%\n' \
                   f'Gender: {"Female" if gender["gender"] else "Male"} - {percent(gender["score"]):.2f}%'
            draw_rect_with_text(*face['face_box'], (0, 255, 0), text)
            for eye in face['eye_boxes']:
                draw_rect(*eye, (0, 0, 255))

        cv2.putText(img, str(int(1 / (perf_counter() - start))), (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), lineType=cv2.LINE_AA)
        cv2.imshow("camera", img)
        if cv2.waitKey(10) == 27:  # Клавиша Esc
            break


if __name__ == '__main__':
    main()

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
import PIL as Image

def create_user(f_id, name):
    web = cv2.VideoCapture(0)
    web.set(3, 640)
    web.set(4, 480)

    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    f_dir = 'dataset'
    path = os.path.join(f_dir, name)

    if not os.path.isdir(path):
        os.makedirs(path)

    counter = 0

    while True:
        ret, img = web.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        multi_face = faces.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in multi_face:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            counter += 1

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            cv2.imwrite(f"{path}/{name}.{f_id}.{counter}.jpg", face)

        cv2.imshow("Image", img)

        k = cv2.waitKey(100) & 0xff

        if k == 27:
            break
        elif counter >= 40:
            break

    web.release()
    cv2.destroyAllWindows()

create_user(1,"Pepcoding")
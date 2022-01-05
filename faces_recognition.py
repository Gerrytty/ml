import cv2, os
import numpy as np
from PIL import Image


def find_faces(path):
    # List of files
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    bb_images = []
    
    for image_path in image_paths:
        img = Image.open(image_path)
        image = np.array(img.convert('L'), 'uint8')
        filename = os.path.split(image_path)[1]

        # Find faces
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            bounding_box_img = cv2.rectangle(np.array(img), (x, y), (x + w, y + h), (0, 255, 0), 5)
            bb_images.append(bounding_box_img)
            cv2.imshow("", bounding_box_img[:,:,::-1])
            cv2.waitKey(1500)

    return bb_images


if __name__ == "__main__":
    cascadePath = "/home/yuliya/cv-lib/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    imgs = find_faces("img/")
# original code by Jack Wotherspoon(https://github.com/theAIGuysCode)
#   - https://github.com/theAIGuysCode/colab-webcam/blob/main/colab_webcam.ipynb

import cv2


def build_haar_cascade_classifier():
    # initialize the Haar Cascade face detection model
    return cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))


def haar_cascade_classifier(model, img):
    """
    Returns:
        faces: 탐지된 얼굴들의 (x,y,w,h) 좌표들
    """
    # grayscale image for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # get face region coordinates
    faces = model.detectMultiScale(gray)

    return faces

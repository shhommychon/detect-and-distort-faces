from detect_faces.deepface import deepface_detector
from detect_faces.opencv import haar_cascade_classifier
from distort_faces.blur import averaging, bilateral_filtering, gaussian_filtering, median_filtering
from distort_faces.color_change import desaturate, invert
from distort_faces.lens_distortion import concave, convex, wave
from distort_faces.mosaic import mosaic
from distort_faces.sketch import sketch

import cv2
import numpy as np
import os

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

face_detect = False
face_detect_model = "none"
face_distort_type = "none"

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    ret, _ = cap.read()
    while ret:
        ret, img = cap.read()

        if face_detect:
            # 아래 옵션 중 얼굴 탐지 모델을 선택하세요.
            if face_detect_model == "cv2_haar":
                faces = haar_cascade_classifier(img)
            elif face_detect_model == "df_cv":
                faces = deepface_detector(img, detector_backend="opencv")
            elif face_detect_model == "df_ssd":
                faces = deepface_detector(img, detector_backend="ssd")
            elif face_detect_model == "df_dlib":
                faces = deepface_detector(img, detector_backend="dlib")
            elif face_detect_model == "df_mc":
                faces = deepface_detector(img, detector_backend="mtcnn")
            elif face_detect_model == "df_ret":
                faces = deepface_detector(img, detector_backend="retinaface")
            elif face_detect_model == "df_mp":
                faces = deepface_detector(img, detector_backend="mediapipe")
            else:
                faces = []

            bbox_array = np.zeros([SCREEN_HEIGHT, SCREEN_WIDTH, 4], dtype=np.uint8)
            
            for x, y, w, h in faces:
                # 아래 옵션 중 얼굴 변형 종류를 선택하세요.
                if face_distort_type == "mosaic":
                    # 모자이크
                    distorted_roi = mosaic(img, x, y, w, h, rate=15)
                elif face_distort_type == "avg":
                    # 일반 블러
                    distorted_roi = averaging(img, x, y, w, h, kernel_size=10)
                elif face_distort_type == "gauss":
                    # 가우시안 블러
                    distorted_roi = gaussian_filtering(img, x, y, w, h, scale=0.1, kernel_size=7)
                elif face_distort_type == "med":
                    # 중앙값 블러
                    distorted_roi = median_filtering(img, x, y, w, h, kernel_size=9)
                elif face_distort_type == "bilat":
                    # 양방향 블러
                    distorted_roi = bilateral_filtering(img, x, y, w, h, scale=0.1, kernel_size=9)
                elif face_distort_type == "wave":
                    # 물결 효과
                    distorted_roi = wave(img, x, y, w, h)
                elif face_distort_type == "convex":
                    # 볼록 렌즈 효과
                    distorted_roi = convex(img, x, y, w, h)
                elif face_distort_type == "concave":
                    # 오목 렌즈 효과
                    distorted_roi = concave(img, x, y, w, h)
                elif face_distort_type == "gray":
                    # 흑백처리
                    distorted_roi = desaturate(img, x, y, w, h)
                elif face_distort_type == "invert":
                    # 색반전
                    distorted_roi = invert(img, x, y, w, h)
                elif face_distort_type == "sk_w":
                    # 스케치 (흰 바탕)
                    distorted_roi = sketch(img, x, y, w, h, sketch_type="white")
                elif face_distort_type == "sk_m":
                    # 스케치 (원본 색과 합침)
                    distorted_roi = sketch(img, x, y, w, h, sketch_type="merged")
                else:
                    distorted_roi = None

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
                img[y:y+h, x:x+w] = cv2.cvtColor(distorted_roi, cv2.COLOR_BGRA2RGB)

        cv2.imshow("Capturing...", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

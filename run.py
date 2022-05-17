from helper.function_keys import detector, distortor

import cv2
import numpy as np

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
            #   "cv2_haar":     cv2.CascadeClassifier
            #   "df_cv":        deepface.detectors.FaceDetector.build_model("opencv")
            #   "df_ssd":       deepface.detectors.FaceDetector.build_model("ssd")
            #   "df_dlib":      deepface.detectors.FaceDetector.build_model("dlib")
            #   "df_mc":        deepface.detectors.FaceDetector.build_model("mtcnn")
            #   "df_ret":       deepface.detectors.FaceDetector.build_model("retinaface")
            #   "df_mp":        deepface.detectors.FaceDetector.build_model("mediapipe")
            faces = detector(face_detect_model, img)

            bbox_array = np.zeros([SCREEN_HEIGHT, SCREEN_WIDTH, 4], dtype=np.uint8)
            
            for x, y, w, h in faces:
                # 아래 옵션 중 얼굴 변형 종류를 선택하세요.
                #   "mosaic":   모자이크
                #   "avg":      일반 블러
                #   "gauss":    가우시안 블러
                #   "med":      중앙값 블러
                #   "bilat":    양방향 블러
                #   "wave":     물결 효과
                #   "convex":   볼록 렌즈 효과
                #   "concave":  오목 렌즈 효과
                #   "gray":     흑백처리
                #   "invert":   색반전
                #   "sk_w":     스케치 (흰 바탕)
                #   "sk_m":     스케치 (원본 색과 합침)
                distorted_roi = distortor(face_distort_type, img, x, y, w, h)

                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
                img[y:y+h, x:x+w] = distorted_roi

        cv2.imshow("Capturing...", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

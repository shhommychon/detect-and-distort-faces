from detect_faces.deepface import build_deepface_detector, deepface_detector
from detect_faces.opencv import build_haar_cascade_classifier, haar_cascade_classifier
from distort_faces.blur import averaging, bilateral_filtering, gaussian_filtering, median_filtering
from distort_faces.color_change import desaturate, invert
from distort_faces.lens_distortion import concave, convex, wave
from distort_faces.mosaic import mosaic
from distort_faces.sketch import sketch


def define_detector(model_type):
    # 아래 옵션 중 얼굴 탐지 모델을 선택하세요.
    if model_type == "cv2_haar":
        model = build_haar_cascade_classifier()
    elif model_type == "df_cv":
        model = build_deepface_detector(detector_backend="opencv")
    elif model_type == "df_ssd":
        model = build_deepface_detector(detector_backend="ssd")
    elif model_type == "df_dlib":
        model = build_deepface_detector(detector_backend="dlib")
    elif model_type == "df_mc":
        model = build_deepface_detector(detector_backend="mtcnn")
    elif model_type == "df_ret":
        model = build_deepface_detector(detector_backend="retinaface")
    elif model_type == "df_mp":
        model = build_deepface_detector(detector_backend="mediapipe")
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    return model


def detector(model, model_type, img):
    # 아래 옵션 중 얼굴 탐지 모델을 선택하세요.
    if model_type == "cv2_haar":
        faces = haar_cascade_classifier(model, img)
    elif model_type == "df_cv":
        faces = deepface_detector(model, img, detector_backend="opencv")
    elif model_type == "df_ssd":
        faces = deepface_detector(model, img, detector_backend="ssd")
    elif model_type == "df_dlib":
        faces = deepface_detector(model, img, detector_backend="dlib")
    elif model_type == "df_mc":
        faces = deepface_detector(model, img, detector_backend="mtcnn")
    elif model_type == "df_ret":
        faces = deepface_detector(model, img, detector_backend="retinaface")
    elif model_type == "df_mp":
        faces = deepface_detector(model, img, detector_backend="mediapipe")
    else:
        faces = []
    
    return faces


def distortor(func_type, img, x, y, w, h, env="my_linux_local"):
    # 아래 옵션 중 얼굴 변형 종류를 선택하세요.
    if func_type == "mosaic":
        # 모자이크
        distorted_roi = mosaic(img, x, y, w, h, rate=15, env=env)
    elif func_type == "avg":
        # 일반 블러
        distorted_roi = averaging(img, x, y, w, h, kernel_size=10, env=env)
    elif func_type == "gauss":
        # 가우시안 블러
        distorted_roi = gaussian_filtering(img, x, y, w, h, scale=0.1, kernel_size=7, env=env)
    elif func_type == "med":
        # 중앙값 블러
        distorted_roi = median_filtering(img, x, y, w, h, kernel_size=9, env=env)
    elif func_type == "bilat":
        # 양방향 블러
        distorted_roi = bilateral_filtering(img, x, y, w, h, scale=0.1, kernel_size=9, env=env)
    elif func_type == "wave":
        # 물결 효과
        distorted_roi = wave(img, x, y, w, h, env=env)
    elif func_type == "convex":
        # 볼록 렌즈 효과
        distorted_roi = convex(img, x, y, w, h, env=env)
    elif func_type == "concave":
        # 오목 렌즈 효과
        distorted_roi = concave(img, x, y, w, h, env=env)
    elif func_type == "gray":
        # 흑백처리
        distorted_roi = desaturate(img, x, y, w, h, env=env)
    elif func_type == "invert":
        # 색반전
        distorted_roi = invert(img, x, y, w, h, env=env)
    elif func_type == "sk_w":
        # 스케치 (흰 바탕)
        distorted_roi = sketch(img, x, y, w, h, sketch_type="white", env=env)
    elif func_type == "sk_m":
        # 스케치 (원본 색과 합침)
        distorted_roi = sketch(img, x, y, w, h, sketch_type="merged", env=env)
    else:
        distorted_roi = None
    
    return distorted_roi
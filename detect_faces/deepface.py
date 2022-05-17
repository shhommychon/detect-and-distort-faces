# original code by Sefik Ilkin Serengil(https://github.com/serengil)
#   - https://github.com/serengil/deepface/blob/master/deepface/detectors/FaceDetector.py#L47-L66

from deepface.detectors import FaceDetector


def build_deepface_detector(detector_backend="opencv"):
    return FaceDetector.build_model(detector_backend)


def deepface_detector(face_detector, img, detector_backend="opencv"):
    """
    Params:
        detector_backend: "opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe" 중 하나
    Returns:
        faces: 탐지된 얼굴들의 (x,y,w,h) 좌표들
    """
    try:
        # faces store list of detected_face and region pair
        faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align=False)
    except:
        return []
    
    return [ face[1] for face in faces ]

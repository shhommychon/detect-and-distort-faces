import cv2

def desaturate(img, x, y, w, h):
    """흑백처리

    Returns:
        roi: 흑백처리 처리된 Region of Interest
    """
    # 관심영역 지정
    roi = img[y:y+h, x:x+w]

    # 흑백으로 변환
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGRA)

    return roi

def invert(img, x, y, w, h):
    """색반전
    
    Returns:
        roi: 색반전 처리된 Region of Interest
    """
    # 관심영역 지정
    roi = img[y:y+h, x:x+w]

    # 색 반전
    roi = cv2.bitwise_not(roi)

    return cv2.cvtColor(roi, cv2.COLOR_RGB2BGRA)

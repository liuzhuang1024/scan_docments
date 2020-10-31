from transform import four_point_transform
import cv2
import imutils
import imEnhance
import numpy as np
import logging 
logging.basicConfig(
    level=logging.INFO
)

print = logging.info

def preProcess(image):
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    edgedImage = cv2.Canny(gaussImage, 75, 200)

    cnts = cv2.findContours(
        edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = ''
    for c in cnts:
        peri = cv2.arcLength(c, True)  # Calculating contour circumference
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    return screenCnt, ratio


if __name__ == "__main__":
    from PIL import Image
    image = cv2.imread("imgs/image.jpg")
    screenCnt, ratio = preProcess(image)
    warped = four_point_transform(image, screenCnt.reshape(4, 2) * ratio)

    # enhancer = imEnhance.Enhancer()
    # enhancedImg = enhancer.gamma(warped, 1.63)

    Image.fromarray(image).show()
    print(warped.shape)
    Image.fromarray(warped).show()

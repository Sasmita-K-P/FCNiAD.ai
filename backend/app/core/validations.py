import cv2
import numpy as np

def validate_image(img, sharpness_thresh=5, brightness_range=(60, 190)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sharpness: variance of Laplacian
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Brightness: mean pixel intensity
    brightness = gray.mean()

    passed = sharpness > sharpness_thresh and brightness_range[0] < brightness < brightness_range[1]
    metrics = {"sharpness": round(sharpness, 2), "brightness": round(brightness, 2)}

    return passed, metrics

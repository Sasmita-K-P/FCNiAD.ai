import cv2
import numpy as np

def extract_nail_roi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Largest contour is likely the hand
    largest_contour = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest_contour)

    # Focus on top area (fingernails region)
    h_crop = int(h * 0.4)
    roi = img[y:y+h_crop, x:x+w]
    return roi

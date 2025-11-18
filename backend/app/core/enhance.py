# app/core/enhance.py
import cv2
import numpy as np

def normalize_color(img, ref_L=125):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mean_L = L.mean()
    if mean_L < 1:
        return img
    scale = ref_L / mean_L
    L2 = np.clip(L * scale, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

def enhance_vascular_contrast(img):
    den = cv2.bilateralFilter(img, 9, 75, 75)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    L2 = clahe.apply(L)

    merged = cv2.merge([L2, A, B])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (0,0), 2.5)
    unsharp = cv2.addWeighted(enhanced, 1.6, blur, -0.6, 0)

    return unsharp

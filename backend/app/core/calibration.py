import cv2
import numpy as np

def calibrate_image(img):
    # Convert to LAB for better color normalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    # Gray-world color balance
    balanced = gray_world_balance(cv2.merge([l_clahe, a, b]))

    calibrated = cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)
    return calibrated

def gray_world_balance(img):
    b, g, r = cv2.split(img)
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg_gray = (avg_b + avg_g + avg_r) / 3

    b = np.clip(b * (avg_gray / avg_b), 0, 255)
    g = np.clip(g * (avg_gray / avg_g), 0, 255)
    r = np.clip(r * (avg_gray / avg_r), 0, 255)

    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])

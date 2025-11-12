import cv2
import numpy as np
from scipy.fft import fft2
import json

def extract_features(img):
    features = {}

    # Mean RGB values
    b,g,r = cv2.split(img)
    features["mean_R"] = float(np.mean(r))
    features["mean_G"] = float(np.mean(g))
    features["mean_B"] = float(np.mean(b))
    features["ratio_RB"] = round((features["mean_R"] / (features["mean_B"] + 1e-5)), 3)

    # LAB a*, b* components (color balance)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    features["lab_a"] = float(np.mean(a))
    features["lab_b"] = float(np.mean(b))

    # FFT texture analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.abs(fft2(gray))
    features["fft_texture"] = round(np.mean(f), 2)

    return features

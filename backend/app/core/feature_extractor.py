import cv2
import numpy as np
from scipy.fft import fft2, fftshift

def extract_features(img):
    """Extract robust spectral + LAB-based color features from ROI."""
    features = {}

    # Mean RGB values
    b, g, r = cv2.split(img)
    features["mean_R"] = float(np.mean(r))
    features["mean_G"] = float(np.mean(g))
    features["mean_B"] = float(np.mean(b))
    features["ratio_RB"] = round((features["mean_R"] / (features["mean_B"] + 1e-5)), 3)

    # LAB color space analysis
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b_lab = cv2.split(lab)
    features["lab_L"] = float(np.mean(l))
    features["lab_a"] = float(np.mean(a))
    features["lab_b"] = float(np.mean(b_lab))
    features["lab_ratio_La"] = round((features["lab_L"] / (features["lab_a"] + 1e-5)), 3)

    # Color skewness (helps detect pallor)
    features["skew_R"] = float(np.std(r))
    features["skew_G"] = float(np.std(g))
    features["skew_B"] = float(np.std(b))

    # FFT-based texture signature (spectral estimation)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.abs(fftshift(fft2(gray)))
    features["fft_energy"] = round(np.mean(f), 2)
    features["fft_contrast"] = round(np.std(f), 2)

    return features

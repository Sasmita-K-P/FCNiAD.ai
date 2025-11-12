# app/core/feature_extractor.py
import cv2, numpy as np, json
from scipy.fft import fft2

def extract_features(img):
    """Extract interpretable features: RGB ratios, LAB, FFT energy."""
    features = {}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    mean_r, mean_g, mean_b = np.mean(img[:,:,2]), np.mean(img[:,:,1]), np.mean(img[:,:,0])
    features["mean_R"], features["mean_G"], features["mean_B"] = float(mean_r), float(mean_g), float(mean_b)
    features["ratio_RB"] = float(mean_r / (mean_b + 1e-6))

    l,a,b = cv2.split(lab)
    features["lab_L"], features["lab_a"], features["lab_b"] = float(np.mean(l)), float(np.mean(a)), float(np.mean(b))

    fft_vals = np.abs(fft2(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    features["fft_mean"], features["fft_std"] = float(np.mean(fft_vals)), float(np.std(fft_vals))
    return features

def fuse_nail_features(rois, fusion_mode="median"):
    """Aggregates features from multiple nails."""
    all_feats = [extract_features(roi) for roi in rois]
    keys = all_feats[0].keys()
    fused = {}
    for k in keys:
        vals = [f[k] for f in all_feats]
        fused[k] = float(np.median(vals) if fusion_mode=="median" else np.mean(vals))
    return fused

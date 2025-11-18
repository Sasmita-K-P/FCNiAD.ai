# app/core/segmentation_unet.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy.ndimage import binary_fill_holes

def load_unet(path):
    return load_model(path, compile=False)

def predict_mask_unet(model, img_bgr, target=(256,256), thresh=0.45):
    ih, iw = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_r = cv2.resize(img_rgb, target) / 255.0
    pred = model.predict(np.expand_dims(img_r,0))[0,:,:,0]

    mask_small = (pred >= thresh).astype(np.uint8)
    mask = cv2.resize(mask_small, (iw,ih), interpolation=cv2.INTER_NEAREST)

    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels <= 1:
        return np.zeros_like(mask)

    max_area = 0; max_label = 1
    for L in range(1, num_labels):
        area = (labels == L).sum()
        if area > max_area:
            max_area = area; max_label = L

    clean = (labels == max_label).astype(np.uint8)
    clean = binary_fill_holes(clean).astype(np.uint8)

    return (clean * 255).astype(np.uint8)

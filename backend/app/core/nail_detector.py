# app/core/nail_detector.py
import cv2
import numpy as np
import mediapipe as mp

from core.segmentation_unet import load_unet, predict_mask_unet
from core.geometry import get_orientation_from_mask, rotate_image_and_mask, rectify_perspective_by_mask, crop_to_mask_bbox
from core.enhance import normalize_color, enhance_vascular_contrast

mp_hands = mp.solutions.hands

# Load UNet if exists
import os
MODEL_PATH = "app/models/unet_best.h5"
USE_UNET = os.path.exists(MODEL_PATH)
UNET_MODEL = load_unet(MODEL_PATH) if USE_UNET else None

def hsv_fallback_mask(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    mask = ((s > 25) & (v > 40) & (s < 200)).astype(np.uint8) * 255
    return mask

def expand_to_nail_bed(image, point):
    cx, cy = point
    ih, iw = image.shape[:2]

    w = 110
    up = 160
    down = 100

    x1 = max(0, cx - w)
    x2 = min(iw, cx + w)
    y1 = max(0, cy - up)
    y2 = min(ih, cy + down)
    return image[y1:y2, x1:x2], (x1,y1,x2,y2)

def detect_nail_roi(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, "Invalid image."

    h, w = img.shape[:2]
    annotated = img.copy()
    rois = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None, annotated, "No hand detected."

        fingertip_ids = [4, 8, 12, 16, 20]

        for hand in res.multi_hand_landmarks:
            for fid in fingertip_ids:
                x = int(hand.landmark[fid].x * w)
                y = int(hand.landmark[fid].y * h)

                big_roi, box = expand_to_nail_bed(img, (x,y))
                bx1,by1,bx2,by2 = box
                cv2.rectangle(annotated,(bx1,by1),(bx2,by2),(0,255,0),2)

                # --- SHAPE-AWARE MASK ---
                if USE_UNET:
                    mask = predict_mask_unet(UNET_MODEL, big_roi)
                else:
                    mask = hsv_fallback_mask(big_roi)

                # rotate
                angle = get_orientation_from_mask(mask)
                rot_img, rot_mask = rotate_image_and_mask(big_roi, mask, angle)

                # rectify
                rect_img, rect_mask = rectify_perspective_by_mask(rot_img, rot_mask)

                # crop
                crop, _ = crop_to_mask_bbox(rect_img, rect_mask)
                if crop is None:
                    continue

                # enhance
                crop = normalize_color(crop)
                crop = enhance_vascular_contrast(crop)

                rois.append(crop)

    return rois, annotated, "Shape-aware ROI extraction completed."

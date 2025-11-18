# app/core/geometry.py
import cv2
import numpy as np

def get_orientation_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0.0
    coords = np.column_stack((xs, ys)).astype(np.float32)
    mean = coords.mean(axis=0)
    cov = np.cov((coords - mean), rowvar=False)
    w, v = np.linalg.eig(cov)
    idx = np.argmax(w)
    main_vec = v[:, idx]
    angle_rad = np.arctan2(main_vec[1], main_vec[0])
    angle_deg = np.degrees(angle_rad)
    return 90.0 - angle_deg  # make major axis vertical

def rotate_image_and_mask(img, mask, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img_r = cv2.warpAffine(img, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderValue=(255, 255, 255))
    mask_r = cv2.warpAffine(mask, M, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderValue=0)
    return img_r, mask_r

def rectify_perspective_by_mask(img, mask, expand=1.15):
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img, mask
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return img, mask

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.float32)

    w = int(rect[1][0] * expand)
    h = int(rect[1][1] * expand)
    if w < 20 or h < 20:
        return img, mask

    def order(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    src = order(box)
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    img_w = cv2.warpPerspective(img, M, (w, h),
                                borderValue=(255,255,255))
    mask_w = cv2.warpPerspective(mask, M, (w, h),
                                 flags=cv2.INTER_NEAREST,
                                 borderValue=0)
    return img_w, mask_w

def crop_to_mask_bbox(img, mask, pad=5):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None, None
    x1 = max(0, xs.min() - pad)
    x2 = min(img.shape[1]-1, xs.max() + pad)
    y1 = max(0, ys.min() - pad)
    y2 = min(img.shape[0]-1, ys.max() + pad)
    return img[y1:y2+1, x1:x2+1], (x1, y1, x2, y2)

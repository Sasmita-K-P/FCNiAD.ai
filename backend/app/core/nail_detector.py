# app/core/nail_detector.py
import cv2
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands

# -----------------------------
# 1. Anatomical Expansion Logic
# -----------------------------
def expand_to_nail_bed(image, fingertip_point, base_w=60, base_h=80):
    """
    Expands ROI to approximate full nail bed:
    - upward: proximal fold + lunula
    - sideways: lateral folds
    - downward: free edge
    """
    ih, iw = image.shape[:2]
    cx, cy = fingertip_point

    w = int(base_w * 2.2)
    h_up = int(base_h * 2.5)
    h_down = int(base_h * 1.0)

    x1 = max(0, cx - w)
    x2 = min(iw, cx + w)
    y1 = max(0, cy - h_up)
    y2 = min(ih, cy + h_down)

    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


# -----------------------------------
# 2. Classical CV Nail-Bed Mask (NO ML)
# -----------------------------------
def generate_nail_mask(roi):
    """
    Classical nail-bed segmentation:
    - LAB threshold for pale-pink region
    - Morph open + close
    - Convex hull = clean nail bed
    """
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Nail bed LAB range (pale pink / beige)
    mask = (L > 120) & (L < 245) & (A > 120) & (A < 158)
    mask = mask.astype(np.uint8) * 255

    # Clean
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Convex hull
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    hull_mask = np.zeros_like(mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, -1)
    return hull_mask


# ---------------------------------------
# 3. Auto-Rotation (PCA on mask)
# ---------------------------------------
def rotate_to_vertical(img, mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return img, mask

    coords = np.column_stack((xs, ys))
    mean = coords.mean(axis=0)
    cov = np.cov(coords - mean, rowvar=False)
    _, eigenvecs = np.linalg.eig(cov)
    main = eigenvecs[:,0]

    angle = np.degrees(np.arctan2(main[1], main[0]))
    rot = 90 - angle

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), rot, 1.0)

    img_r = cv2.warpAffine(img, M, (w,h), borderValue=(255,255,255))
    mask_r = cv2.warpAffine(mask, M, (w,h))
    return img_r, mask_r


# ---------------------------------------
# 4. Perspective correction
# ---------------------------------------
def rectify_perspective(img, mask):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.float32)

    w = int(rect[1][0])
    h = int(rect[1][1])

    dst = np.array([[0,h-1],[0,0],[w-1,0],[w-1,h-1]],dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)

    warped = cv2.warpPerspective(img, M, (w,h))
    return warped


# ---------------------------------------
# 5. Enhancement
# ---------------------------------------
def enhance_nail(img):
    den = cv2.bilateralFilter(img, 9, 75, 75)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)

    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    final = cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)
    return final


# ---------------------------------------
# 6. MAIN EXTRACTOR
# ---------------------------------------
def detect_nail_roi(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, "Invalid image"

    h,w = img.shape[:2]
    annotated = img.copy()
    rois = []

    with mp_hands.Hands(static_image_mode=True,max_num_hands=1) as hands:
        res = hands.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

        if not res.multi_hand_landmarks:
            return None, img, "No hand detected"

        for hand in res.multi_hand_landmarks:
            # Fingertip IDs
            tips = [4,8,12,16,20]
            for fid in tips:
                x = int(hand.landmark[fid].x * w)
                y = int(hand.landmark[fid].y * h)

                # 1) Anatomical expansion
                big_roi, box = expand_to_nail_bed(img, (x,y))
                bx1,by1,bx2,by2 = box
                cv2.rectangle(annotated,(bx1,by1),(bx2,by2),(0,255,0),2)

                # 2) Mask
                mask = generate_nail_mask(big_roi)

                # 3) Rotate
                rot_img, rot_mask = rotate_to_vertical(big_roi, mask)

                # 4) Perspective fix
                rectified = rectify_perspective(rot_img, rot_mask)

                # 5) Enhance
                enhanced = enhance_nail(rectified)

                rois.append(enhanced)

    if len(rois)==0:
        return None, annotated, "No ROI extracted"

    return rois, annotated, "Full nail-bed extracted successfully"

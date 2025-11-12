# app/core/nail_detector.py
import cv2, numpy as np, mediapipe as mp
mp_hands = mp.solutions.hands

def detect_nail_roi(image_path, debug=False):
    """
    Detects nail ROI from finger or hand image.
    Rejects painted nails or invalid inputs.
    Returns ([list of ROIs], annotated_image, status_msg)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, "❌ Invalid image path."

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    annotated = img.copy()
    rois = []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.4, model_complexity=1) as hands:
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            # 🔁 Fallback: contour-based detection for finger-only images
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blur, 40, 120)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500 or area > 5000:
                    continue
                x, y, w_, h_ = cv2.boundingRect(cnt)
                aspect_ratio = w_ / float(h_ + 1e-6)
                # Nail-like contour near top region
                if 0.4 < aspect_ratio < 1.2 and y < img.shape[0] // 2:
                    roi = img[y:y + h_, x:x + w_]
                    if roi.size == 0:
                        continue
                    if is_painted_nail(roi):
                        return None, annotated, "💅 Painted nail detected. Please remove nail polish."
                    if is_skin_color_region(roi):
                        rois.append(roi)
                        cv2.rectangle(annotated, (x, y), (x + w_, y + h_), (0, 255, 0), 2)

            if len(rois) == 0:
                return None, annotated, "❌ No valid nail found (finger-only image not detected properly)."


        for hand_landmarks in results.multi_hand_landmarks:
            fingertip_ids = [4, 8, 12, 16, 20]
            for fid in fingertip_ids:
                x = int(hand_landmarks.landmark[fid].x * w)
                y = int(hand_landmarks.landmark[fid].y * h)
                cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)
                nail_h, nail_w = 80, 100
                y1, y2 = max(0, y - nail_h), min(h, y)
                x1, x2 = max(0, x - nail_w // 2), min(w, x + nail_w // 2)
                roi = img[y1:y2, x1:x2]
                if roi.size == 0: continue
                if is_painted_nail(roi):
                    return None, annotated, "💅 Painted nail detected. Please remove nail polish."
                if is_skin_color_region(roi) :
                    rois.append(roi)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if len(rois) == 0:
        return None, annotated, "❌ No valid nail region found."
    return rois, annotated, "✅ Nail ROI(s) extracted successfully."

def is_skin_color_region(roi):
    """Checks whether the detected region resembles skin tone."""
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    skin_mask = (A > 120) & (A < 160) & (B > 120) & (B < 160)
    ratio = np.sum(skin_mask) / (roi.shape[0] * roi.shape[1])
    return ratio > 0.4  # At least 40% must be skin

def is_painted_nail(roi):
    """Detects painted nails via high HSV saturation."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_sat, mean_val = np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])
    return mean_sat > 85 and mean_val > 130

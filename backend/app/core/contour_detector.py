import cv2
import numpy as np
from core.nail_detector import expand_to_nail_bed

def extract_nail_roi(img):
    """Detect and extract the nail region using skin masking + contour filtering."""
    original = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step 1: Skin color range (broad range for various tones)
    lower = np.array([0, 15, 60], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Step 2: Morphological cleanup
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Step 3: Contour detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("⚠️ No skin-like region found.")
        return None

    # Step 4: Sort by area, pick top few contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    roi = None
    best_box = None
    img_h, img_w = img.shape[:2]
    top_section = int(img_h * 0.45)  # nails are in the upper half of hand

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if y < top_section and w > 30 and h > 20:
            # Expand contour box to nail-bed proportions
            nx1, ny1, nx2, ny2 = expand_to_nail_bed(x, y, w, h, img_w, img_h)
            roi = img[ny1:ny2, nx1:nx2]

            best_box = (x, y, w, h)
            break

    if roi is None:
        print("⚠️ No valid nail contour detected.")
        return None

    # Step 5: Draw green rectangle for visualization
    cv2.rectangle(original, (best_box[0], best_box[1]),
                  (best_box[0]+best_box[2], best_box[1]+best_box[3]),
                  (0, 255, 0), 3)

    return roi, original

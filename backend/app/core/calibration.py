# app/core/calibration.py
import cv2
import numpy as np

# ============================================================
# 1️⃣ Brightness Normalization (Ethical Fairness Step)
# ============================================================
def normalize_brightness(img):
    """
    Normalizes overly bright or dark regions in LAB color space.
    Ensures fairness across all skin tones by stabilizing luminance (L*).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mean_L = np.mean(L)
    # Overexposed images (too bright)
    if mean_L > 200:
        L = cv2.equalizeHist(np.clip(L, 0, 200).astype(np.uint8))
    # Underexposed / dark lighting
    elif mean_L < 50:
        L = cv2.equalizeHist(np.clip(L, 50, 255).astype(np.uint8))

    lab2 = cv2.merge([L, A, B])
    normalized = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return normalized


# ============================================================
# 2️⃣ White-Patch Calibration
# ============================================================
def white_patch_balance(img):
    """
    White-Patch algorithm: assumes the brightest pixel should be pure white.
    Helps correct global color cast (useful for yellow or bluish lighting).
    """
    img_float = img.astype(np.float32)
    max_per_channel = np.percentile(img_float.reshape(-1, 3), 99, axis=0)
    scale = 255.0 / (max_per_channel + 1e-6)
    balanced = img_float * scale
    return np.clip(balanced, 0, 255).astype(np.uint8)


# ============================================================
# 3️⃣ Gray-World Normalization
# ============================================================
def gray_world_balance(img):
    """
    Implements the Gray-World Assumption:
    The average color of a natural image should be neutral gray.
    """
    b, g, r = cv2.split(img.astype(np.float32))
    avg_b, avg_g, avg_r = b.mean(), g.mean(), r.mean()
    avg_gray = (avg_b + avg_g + avg_r) / 3.0 + 1e-6

    b *= avg_gray / avg_b
    g *= avg_gray / avg_g
    r *= avg_gray / avg_r

    return cv2.merge([
        np.clip(b, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(r, 0, 255).astype(np.uint8)
    ])


# ============================================================
# 4️⃣ Local Contrast Enhancement (CLAHE)
# ============================================================
def apply_clahe(img):
    """
    Enhances local contrast to highlight subtle pallor changes.
    Works in LAB color space to preserve hue information.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


# ============================================================
# 5️⃣ Master Calibration Function (Full Pipeline)
# ============================================================
def calibrate_image(img):
    """
    Full spectral calibration pipeline:
    1. Brightness normalization
    2. White-patch color cast correction
    3. Gray-world normalization
    4. CLAHE local enhancement
    """
    step1 = normalize_brightness(img)
    step2 = white_patch_balance(step1)
    step3 = gray_world_balance(step2)
    step4 = apply_clahe(step3)
    return step4


# ============================================================
# 6️⃣ Debug Visualization (Optional, for testing fairness)
# ============================================================
def visualize_calibration_stages(original):
    """
    Debug utility: visualize each stage in one window.
    Helps verify that tone normalization and brightness correction are effective.
    """
    wp = white_patch_balance(original)
    gw = gray_world_balance(wp)
    clahe = apply_clahe(gw)
    norm = normalize_brightness(clahe)

    combined = np.hstack([
        cv2.resize(original, (200, 200)),
        cv2.resize(wp, (200, 200)),
        cv2.resize(gw, (200, 200)),
        cv2.resize(clahe, (200, 200)),
        cv2.resize(norm, (200, 200)),
    ])

    cv2.imshow("Calibration Stages: Original → WP → GW → CLAHE → Brightness Norm", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# app/main.py
"""
ROBUST & FAIR DATA PREPROCESSING BACKEND
----------------------------------------
Performs:
  1. Image quality validation
  2. Nail ROI detection (single/multi-finger or hand)
  3. CLAHE + Gray-World calibration
  4. Fairness & skin-tone normalization (LAB)
  5. Spectral feature extraction
Outputs:
  - Annotated result
  - Calibrated ROI(s)
  - Fairness report (JSON)
  - Extracted features (JSON)
Author: GPT-5 (COE-Intern Ready Version)
"""

import os, cv2, json
from pathlib import Path
from utils.logger import info, success, warn, error
from utils.file_manager import ensure_folders, clean_output
from core.nail_detector import detect_nail_roi
from core.calibration import calibrate_image
from core.feature_extractor import fuse_nail_features
from core.tone_fairness import analyze_tone_fairness

# Folder configuration
BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE / "input"
OUTPUT_DIR = BASE / "output"
ensure_folders([INPUT_DIR, OUTPUT_DIR])

def save_json(data, filename):
    """Helper: save any Python dict as a formatted JSON file"""
    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(data, f, indent=4)

def main():
    info("🚀 Starting Robust & Fair Preprocessing Pipeline...")

    # --- Step 1: Input Validation ---
    input_images = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not input_images:
        warn("No image found in input/. Please place one hand or finger image and rerun.")
        return
    img_path = input_images[-1]
    info(f"📸 Using image: {img_path.name}")

    clean_output(OUTPUT_DIR)

    # --- Step 2: Nail ROI Detection ---
    info("🔍 Detecting nail regions...")
    result = detect_nail_roi(str(img_path), debug=False)

    if result is None:
        error("❌ Internal detector failure. Check nail_detector.py.")
        return

    rois, annotated, status = result
    info(status)

    if rois is None or len(rois) == 0:
        error("❌ No valid ROI detected. Please check lighting, clarity, or remove nail polish.")
        cv2.imwrite(str(OUTPUT_DIR / "failed_result.jpg"), annotated)
        return

    cv2.imwrite(str(OUTPUT_DIR / "annotated_result.jpg"), annotated)
    success(f"✅ ROI detection completed. {len(rois)} region(s) found.")

    # --- Step 3: Fairness-Calibrated Preprocessing ---
    info("🎨 Applying CLAHE + Gray-World Calibration...")
    calibrated_rois = [calibrate_image(r) for r in rois]

    for i, r in enumerate(calibrated_rois):
        cv2.imwrite(str(OUTPUT_DIR / f"calibrated_roi_{i+1}.jpg"), r)
    success("✅ Calibration done and saved.")

    # --- Step 4: Tone Fairness & Skin Normalization ---
    info("🌈 Performing skin-tone fairness analysis...")
    fairness_report = analyze_tone_fairness(calibrated_rois)
    save_json(fairness_report, "fairness_report.json")
    success("✅ Fairness normalization and audit complete.")

    # --- Step 5: Feature Extraction (Spectral + LAB + HSV + FFT) ---
    info("📊 Extracting interpretable spectral features...")
    fused_features = fuse_nail_features(calibrated_rois, fusion_mode="median")
    save_json(fused_features, "features.json")
    success("✅ Feature extraction complete and saved to features.json")

    # --- Step 6: Summary Output ---
    info("\n============================")
    info("🏁 PIPELINE SUMMARY")
    info("============================")
    success(f"Input Image: {img_path.name}")
    success(f"Detected ROIs: {len(rois)}")
    success(f"Output Folder: {OUTPUT_DIR}")
    info("Files Generated:")
    print("  - annotated_result.jpg")
    print("  - calibrated_roi_X.jpg")
    print("  - fairness_report.json")
    print("  - features.json")

    info("🎯 Robust preprocessing completed successfully!")
    info("Use 'features.json' + 'fairness_report.json' for training or audit phase.")

if __name__ == "__main__":
    main()

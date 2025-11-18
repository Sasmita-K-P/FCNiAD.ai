# app/main.py
import os, cv2, json
from pathlib import Path
from utils.logger import info, success, warn, error
from utils.file_manager import ensure_folders, clean_output

from core.nail_detector import detect_nail_roi
from core.calibration import calibrate_image
from core.feature_extractor import fuse_nail_features
from core.tone_fairness import analyze_tone_fairness

BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE/"input"
OUTPUT_DIR = BASE/"output"
ensure_folders([INPUT_DIR, OUTPUT_DIR])

def save_json(d, name):
    with open(OUTPUT_DIR/name, "w") as f:
        json.dump(d, f, indent=4)

def main():
    info("🚀 Starting Shape-Aware Nail-Bed Extraction Pipeline...")

    imgs = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in(".jpg",".png")]
    if not imgs:
        warn("Place an image inside input/.")
        return

    img_path = imgs[-1]
    clean_output(OUTPUT_DIR)

    rois, annotated, status = detect_nail_roi(str(img_path))
    info(status)

    if not rois:
        error("❌ No ROIs extracted.")
        cv2.imwrite(str(OUTPUT_DIR/"failed.jpg"), annotated)
        return

    cv2.imwrite(str(OUTPUT_DIR/"annotated_result.jpg"), annotated)

    final_rois = rois[:5]

    for i,r in enumerate(final_rois,1):
        calibrated = calibrate_image(r)
        cv2.imwrite(str(OUTPUT_DIR/f"calibrated_roi_{i}.jpg"), calibrated)

    fused = fuse_nail_features(final_rois)
    fairness = analyze_tone_fairness(final_rois)

    save_json(fused, "features.json")
    save_json(fairness, "fairness_report.json")

    success("🎉 Done. ROI Extraction + Preprocessing Completed.")

if __name__=="__main__":
    main()

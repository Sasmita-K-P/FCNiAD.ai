import cv2
import os
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from core.validations import validate_image
from core.contour_detector import extract_nail_roi
from core.calibration import calibrate_image
from core.feature_extractor import extract_features
from utils.file_manager import prepare_folders, get_latest_input_image, clear_output_folder
from utils.logger import log_step, log_success, log_error

# ========== PATH CONFIG ==========
INPUT_DIR = Path("app/input")
OUTPUT_DIR = Path("app/output")

def visualize_stages(original, roi, calibrated, save_path):
    """Creates a side-by-side visualization for demo purposes."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axs[1].set_title("ROI (Nail Region)")
    axs[2].imshow(cv2.cvtColor(calibrated, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Calibrated Image")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    log_step("🚀 Starting Preprocessing & Feature Extraction Pipeline...")

    prepare_folders(INPUT_DIR, OUTPUT_DIR)
    clear_output_folder(OUTPUT_DIR)

    img_path = get_latest_input_image(INPUT_DIR)
    if not img_path:
        log_error("❌ No image found in input folder.")
        return

    img = cv2.imread(str(img_path))
    if img is None:
        log_error("❌ Failed to read the image file.")
        return

    # Progress bar for demo
    steps = ["Validate Image", "Extract ROI", "Calibrate Colors", "Extract Features", "Visualize Results"]
    for step in tqdm(steps, desc="Pipeline Progress", ncols=80):

        if step == "Validate Image":
            passed, metrics = validate_image(img)
            if not passed:
                log_error(f"Image failed validation: {metrics}")
                return
            log_success(f"✅ Validation passed: {metrics}")

        elif step == "Extract ROI":
            result = extract_nail_roi(img)
            if result is None:
                log_error("❌ ROI extraction failed.")
                return

            roi, outlined_img = result
            roi_path = OUTPUT_DIR / f"{img_path.stem}_roi.jpg"
            outlined_path = OUTPUT_DIR / f"{img_path.stem}_outlined.jpg"
            cv2.imwrite(str(roi_path), roi)
            cv2.imwrite(str(outlined_path), outlined_img)
            log_success(f"✂️ ROI extracted and outlined: {outlined_path.name}")

        elif step == "Calibrate Colors":
            calibrated = calibrate_image(roi)
            calibrated_path = OUTPUT_DIR / f"{img_path.stem}_calibrated.jpg"
            cv2.imwrite(str(calibrated_path), calibrated)

        elif step == "Extract Features":
            features = extract_features(calibrated)
            feature_path = OUTPUT_DIR / f"{img_path.stem}_features.json"
            with open(feature_path, "w") as f:
                json.dump(features, f, indent=4)

        elif step == "Visualize Results":
            vis_path = OUTPUT_DIR / f"{img_path.stem}_comparison.jpg"
            visualize_stages(img, roi, calibrated, vis_path)
            log_success(f"🖼️ Visualization saved: {vis_path.name}")

    log_success("🎉 Preprocessing & Feature Extraction Completed Successfully!")

if __name__ == "__main__":
    main()

import os
from pathlib import Path

def prepare_folders(input_dir, output_dir):
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

def clear_output_folder(output_dir):
    for f in output_dir.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass

def get_latest_input_image(input_dir):
    images = sorted(input_dir.glob("*.[jp][pn]g"), key=os.path.getmtime)
    return images[-1] if images else None

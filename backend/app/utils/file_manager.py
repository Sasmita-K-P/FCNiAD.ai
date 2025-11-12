# app/utils/file_manager.py
import os, shutil

def ensure_folders(folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def clean_output(output_dir):
    """Clears all previous output images and JSONs before new run."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return
    for f in os.listdir(output_dir):
        fp = os.path.join(output_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)
        elif os.path.isdir(fp):
            shutil.rmtree(fp)

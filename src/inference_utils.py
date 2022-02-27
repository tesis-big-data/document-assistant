import os
import shutil
from pathlib import Path
import subprocess

from constants import (
    INFERENCE_CURRENT_EXEC_CROPPED_IMAGES_PATH,
    INFERENCE_CURRENT_EXEC_JSON_PATH,
    INFERENCE_CURRENT_EXEC_RAW_IMAGES_PATH,
    INFERENCE_CURRENT_EXEC_ROTATED_IMAGES_PATH,
    INFERENCE_HISTORIC_CROPPED_IMG_PATH,
    INFERENCE_HISTORIC_JSON_PATH,
    INFERENCE_HISTORIC_RAW_IMG_PATH,
    INFERENCE_HISTORIC_ROTATED_IMG_PATH,
)


def dvc_push_data():
    subprocess.run(["dvc", "push"])


def remove_folder_files(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def move_files(from_dir, to_dir):
    Path(to_dir).mkdir(exist_ok=True, parents=True)
    file_names = os.listdir(from_dir)

    for file_name in file_names:
        shutil.move(os.path.join(from_dir, file_name), to_dir)


# save execution in historic folder
def save_processing():
    folders_to_historic = [
        {"from": INFERENCE_CURRENT_EXEC_JSON_PATH, "to": INFERENCE_HISTORIC_JSON_PATH},
        {
            "from": INFERENCE_CURRENT_EXEC_RAW_IMAGES_PATH,
            "to": INFERENCE_HISTORIC_RAW_IMG_PATH,
        },
        {
            "from": INFERENCE_CURRENT_EXEC_ROTATED_IMAGES_PATH,
            "to": INFERENCE_HISTORIC_ROTATED_IMG_PATH,
        },
        {
            "from": INFERENCE_CURRENT_EXEC_CROPPED_IMAGES_PATH,
            "to": INFERENCE_HISTORIC_CROPPED_IMG_PATH,
        },
    ]
    for obj in folders_to_historic:
        move_files(obj["from"], obj["to"])

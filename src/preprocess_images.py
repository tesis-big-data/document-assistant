import cv2
from pathlib import Path
import shutil
import uuid
import os
import simplejson as json
from constants import (
    INFERENCE_RAW_IMAGES_PATH,
    INFERENCE_CURRENT_EXEC_JSON_PATH,
    INFERENCE_CURRENT_EXEC_RAW_IMAGES_PATH,
    INFERENCE_CURRENT_EXEC_ROTATED_IMAGES_PATH,
)
from cv_utils import (
    adjust_image_size,
    auto_adjust_image_rotation,
    adjust_image_orientation,
    get_ocr_data,
)

# Accepted file extensions to be processed
valid_file_extensions = [".jpg", ".jpeg", ".png"]


def preprocess_images():
    # Walk into directory looking for files to process
    for root, _, files in os.walk(INFERENCE_RAW_IMAGES_PATH, topdown=False):
        for filename in files:
            extension = os.path.splitext(filename)[1]
            name = os.path.splitext(filename)[0]
            folder = root.split("\\")[-1]

            print(extension)
            print(name)
            print(folder)

            if extension not in valid_file_extensions:
                print(f"{filename} invalid extension, ignored.")
                continue

            # Save filename with uuid
            uuid_filename = str(uuid.uuid4())
            raw_image_path = (
                f"{INFERENCE_CURRENT_EXEC_RAW_IMAGES_PATH}/{uuid_filename}{extension}"
            )

            # Moves image with new uuid name
            os.rename(
                f"{folder}/{name}{extension}",
                raw_image_path,
            )

            # JSON file path
            image_json_path = f"{INFERENCE_CURRENT_EXEC_JSON_PATH}/{uuid_filename}.json"

            # Fixed image path
            processed_image_path = f"{INFERENCE_CURRENT_EXEC_ROTATED_IMAGES_PATH}/processed_{uuid_filename}{extension}"

            # Process image if processed img doesn't exist
            if not Path(processed_image_path).is_file():
                print(" ")
                print("Processing...    " + raw_image_path)
                img_data = {}
                img_data["name"] = uuid_filename
                img_data["extension"] = str(extension)
                img_data["document_id"] = uuid_filename

                # Get the file to be processed
                print("Loading Image...")
                processed_image = cv2.imread(raw_image_path)

                # Adjust Image size to match Max Size
                print("Resizing Image...")
                processed_image, aux_img_data = adjust_image_size(processed_image)
                img_data.update(aux_img_data)

                # Adjust Image rotation to straighten lines
                print("Rotating Image...")
                (
                    processed_image,
                    img_data["img_correction_angle"],
                ) = auto_adjust_image_rotation(processed_image)

                # Adjust Image so text is legible
                print("Text Orienting Image...")
                (
                    processed_image,
                    img_data["img_orientation"],
                ) = adjust_image_orientation(processed_image)

                # Add the Orientation to the correction angle
                img_data["img_correction_angle"] = (
                    img_data["img_correction_angle"] - img_data["img_orientation"]
                )

                # Adjust Image so text is legible
                print("Extracting OCR from Image...")
                processed_image, img_data["OCR_Data"] = get_ocr_data(processed_image)

                # Save proccessed file in same location
                print("Saving Processed Image...")
                cv2.imwrite(processed_image_path, processed_image)

                # Save data file
                print("Saving Processed Image Data (JSON file)...")
                image_json_file = open(image_json_path, "w")
                image_json_file.write(json.dumps(img_data, indent=4, sort_keys=True))
                image_json_file.close()


if __name__ == "__main__":
    Path(INFERENCE_CURRENT_EXEC_JSON_PATH).mkdir(exist_ok=True, parents=True)
    Path(INFERENCE_CURRENT_EXEC_RAW_IMAGES_PATH).mkdir(exist_ok=True, parents=True)
    Path(INFERENCE_CURRENT_EXEC_ROTATED_IMAGES_PATH).mkdir(exist_ok=True, parents=True)
    preprocess_images()

import io
import cv2
import os
import pandas as pd
import simplejson as json
import cv_utils
import time

from constants import (
    INFERENCE_CURRENT_EXEC_CROPPED_IMAGES_PATH,
    INFERENCE_CURRENT_EXEC_ROTATED_IMAGES_PATH,
    INFERENCE_CURRENT_EXEC_PATH,
    INFERENCE_CURRENT_EXEC_JSON_PATH,
    INFERENCE_TEMPLATES_PATH,
)

# Accepted file extensions to be processed
accepted_file_extension = [".jpg", ".jpeg", ".png"]


def extract_fields():
    classified_df = pd.read_parquet(f"{INFERENCE_CURRENT_EXEC_PATH}/classified.parquet")
    # Walk into directory looking for files to process
    for root, _, files in os.walk(
        INFERENCE_CURRENT_EXEC_ROTATED_IMAGES_PATH, topdown=False
    ):
        for filename in files:
            extension = os.path.splitext(filename)[1]
            name = os.path.splitext(filename)[0]
            folder = root.split("\\")[-1]

            document_id = name.replace("processed_", "")
            document_id = document_id.replace(".png", "")

            client = classified_df[classified_df["document_id"] == document_id][
                "cliente"
            ].values[0]
            ocr_text_paragraph = classified_df[
                classified_df["document_id"] == document_id
            ]["OCR_text"].values[0]

            with open(
                f"{INFERENCE_CURRENT_EXEC_JSON_PATH}/{document_id}.json", "r"
            ) as json_file:
                json_document = json.load(json_file)

            client_template = f"{INFERENCE_TEMPLATES_PATH}/{client}"
            template = {
                "image": f"{client_template}/{client}_template.png",
                "json": f"{client_template}/{client}_template.json",
            }

            rotated_image_path = f"{folder}/{name}{extension}"
            cropped_image_path = f"{INFERENCE_CURRENT_EXEC_CROPPED_IMAGES_PATH}/cropped_{document_id}{extension}"

            if extension in accepted_file_extension and not os.path.isfile(
                cropped_image_path
            ):
                print(" ")
                print("Processing...    " + filename)

                # Template json
                template_data = json.load(open(template["json"]))

                # Get the file to be processed
                print("Loading Image...")
                img_to_be_processed = cv2.imread(os.path.join(root, filename))
                template_to_be_processed = cv2.imread(template["image"])

                # Adjust Image so text is legible
                print("Matching Template on Image...")
                img_to_be_processed, matching_result = cv_utils.template_match(
                    template_to_be_processed, img_to_be_processed
                )

                if matching_result:
                    # Adjust Image so text is legible
                    print("Extracting OCR from Image...")

                    is_success, im_buf_arr = cv2.imencode(".png", img_to_be_processed)
                    bytes_img_to_be_processed = io.BytesIO(im_buf_arr)
                    # print(type(bytes_img_to_be_processed))

                    (
                        img_to_be_processed,
                        json_document["OCR_Data_Crop"],
                    ) = cv_utils.get_ocr_data_azure(
                        bytes_img_to_be_processed, img_to_be_processed
                    )

                    # Extract data from image
                    print("Extracting data OCR...")
                    json_document = cv_utils.get_ocr_data_keypoints(
                        img_to_be_processed,
                        template_data,
                        json_document,
                        retry_ocr=False,
                    )

                    # Save proccessed file in same location
                    print("Saving Cropped Image...")
                    cv2.imwrite(cropped_image_path, img_to_be_processed)

                    # Save data file
                    print("Saving Processed Image Data...")
                    json_document["OCR_text_paragraph"] = ocr_text_paragraph
                    json_document["client"] = client

                    with open(
                        f"{INFERENCE_CURRENT_EXEC_JSON_PATH}/{document_id}.json", "w"
                    ) as json_file:
                        json.dump(json_document, json_file)

                else:
                    # Template no detectado
                    print("Template no detectado en Image Data...")
        time.sleep(2)

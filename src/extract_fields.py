import io
import cv2
import os
import pandas as pd
import simplejson as json
import cv_utils
import time
import ex

from constants import (
    INFERENCE_CURRENT_EXEC_CROPPED_IMAGES_PATH,
    INFERENCE_CURRENT_EXEC_ROTATED_IMAGES_PATH,
    INFERENCE_CURRENT_EXEC_PATH,
    INFERENCE_CURRENT_EXEC_JSON_PATH,
    INFERENCE_TEMPLATES_PATH,
)

# Accepted file extensions to be processed
accepted_file_extension = [".jpg", ".jpeg", ".png"]


def extract_fields(uuid_filename):
    #classified_df = pd.read_parquet(f"{INFERENCE_CURRENT_EXEC_PATH}/classified.parquet")
    document_id = uuid_filename

    #client = classified_df[classified_df["document_id"] == document_id][
    #    "cliente"
    #].values[0]
    client = "Aldo"
    #ocr_text_paragraph = classified_df[
    #    classified_df["document_id"] == document_id
    #]["OCR_text"].values[0]
    ocr_text_paragraph = "HOLA"

    with open(
        f"{INFERENCE_CURRENT_EXEC_JSON_PATH}/{document_id}.json", "r"
    ) as json_file:
        json_document = json.load(json_file)

    client_template = f"{INFERENCE_TEMPLATES_PATH}/{client}"
    template = {
        "image": f"{client_template}/{client}_template.png",
        "json": f"{client_template}/{client}_template.json",
    }

    cropped_image_path = f"{INFERENCE_CURRENT_EXEC_CROPPED_IMAGES_PATH}/cropped_{document_id}.png"
    img_to_be_processed_path = f"{INFERENCE_CURRENT_EXEC_ROTATED_IMAGES_PATH}/processed_{document_id}.png"

    if not os.path.isfile(cropped_image_path):
        print(" ")
        print("Processing...    " + document_id)

        # Template json
        template_data = json.load(open(template["json"]))

        # Get the file to be processed
        print("Loading Image...")
        img_to_be_processed = cv2.imread(img_to_be_processed_path)
        template_to_be_processed = cv2.imread(template["image"])

        # Adjust Image so text is legible
        print("Matching Template on Image...")
        img_to_be_processed, matching_result = cv_utils.template_match(
            template_to_be_processed, img_to_be_processed
        )

        # Save proccessed file in same location
        print("Saving Cropped Image...")
        cv2.imwrite(cropped_image_path, img_to_be_processed)

        if matching_result:
            # Adjust Image so text is legible
            print("Extracting OCR from Image...")

            is_success, im_buf_arr = cv2.imencode(".png", img_to_be_processed)
            bytes_img_to_be_processed = io.BytesIO(im_buf_arr)
            
            json_document["OCR"]["Extraction_OCR"] = {}

            (
                img_to_be_processed,
                json_document ["OCR"]["Extraction_OCR"]
            ) = cv_utils.get_ocr_data_azure(
                bytes_img_to_be_processed, img_to_be_processed
            )

            json_document["Extracted_Data"] = {}
        
            # Extract data from image
            print("Extracting data OCR...")
            json_document = cv_utils.get_ocr_data_keypoints(
                img_to_be_processed,
                template_data,
                json_document,
                retry_ocr=False,
            )

            # Draw results into cropped img
            txt = f"Client: {client} - Fecha: {json_document['Extracted_Data']['Fecha']} - Numero: {json_document['Extracted_Data']['Numero']} - Total: {json_document['Extracted_Data']['Total']}"
            cv_utils.draw_text(img_to_be_processed, txt, pos=(40, 20))

            # Save proccessed file in same location
            print("Saving Cropped Image...")
            cv2.imwrite(cropped_image_path, img_to_be_processed)

            # Save data file
            print("Saving Processed Image Data...")
            json_document["OCR"]["Extraction_OCR"]["OCR_text_paragraph"] = ocr_text_paragraph
            json_document["Extracted_Data"]["Classified_Client"] = client

            with open(
                f"{INFERENCE_CURRENT_EXEC_JSON_PATH}/{document_id}.json", "w"
            ) as json_file:
                json.dump(json_document, json_file, indent=4, sort_keys=True)

        else:
            # Template no detectado
            print("Template no detectado en Image Data...")

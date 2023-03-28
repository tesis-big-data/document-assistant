import cv2
from pathlib import Path
import shutil
import uuid
import os
import simplejson as json
from client_export_utils import gdrive_SpreadSheet_open
from constants import (
    GDRIVE_AUTH_KEYS_PATH,
    INFERENCE_CURRENT_EXEC_JSON_PATH
)

def fsotte(document_id):
    try:
        print("Processing...    " + document_id)

        with open(
            f"{INFERENCE_CURRENT_EXEC_JSON_PATH}/{document_id}.json", "r"
        ) as json_file:
            json_document = json.load(json_file)

        if "Classified_Client" in json_document.keys() and json_document["Classified_Client"] == "Los Nietitos":
            gdrive_client = gdrive_SpreadSheet_open(GDRIVE_AUTH_KEYS_PATH)

            open_client=gdrive_client.open("TABLA PRUEBA")

            json_document["Cliente"]
            json_document["Fecha"]
            json_document["Numero"]
            json_document["Total"]


    except:
        print ("Error al abrir google sheets")


if __name__ == "__main__":
    fsotte()
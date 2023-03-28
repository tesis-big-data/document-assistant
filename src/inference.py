from preprocess_images import preprocess_images
from classification import classify
from extract_fields import extract_fields
from inference_utils import save_processing, dvc_push_data
from client_export import fsotte
from constants import (
    INFERENCE_RAW_IMAGES_PATH,
    INFERENCE_CURRENT_EXEC_JSON_PATH,
    INFERENCE_CURRENT_EXEC_RAW_IMAGES_PATH,
    INFERENCE_CURRENT_EXEC_ROTATED_IMAGES_PATH,
)
import os

if __name__ == "__main__":
    #fsotte("Prueba")
    for root, _, files in os.walk(INFERENCE_RAW_IMAGES_PATH, topdown=False):
        for filename in files:
            #Preparo la imagen corrijo orientación 
            uuid_filename,uuid_data = preprocess_images(root,files,filename)

            #Clasifico el documento
            #classify(uuid_filename)

            #Aplico Template y extraigo los datos
            extract_fields(uuid_filename)

            #Extraigo QR
            #extract_qr(uuid_filename)

            #Exporto datos para empresa asociada
            #fsotte(uuid_filename)

            #Guardo archivos en el histórico
            #save_processing()

            # dvc_push_data()

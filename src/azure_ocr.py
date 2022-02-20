from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from environs import Env
from pathlib import Path
from typing import List
import time

env = Env()
env.read_env()


class AzureOCR:
    def __init__(self):
        self.azure_ocr_client = self._authenticate_azure_ocr()

    """
    Authenticate
    Authenticates your credentials and creates a client.
    """

    def _authenticate_azure_ocr(self):
        subscription_key = "33c11e77685b420aa4819823958cad56"
        endpoint = "https://ort.cognitiveservices.azure.com/"

        return ComputerVisionClient(
            endpoint, CognitiveServicesCredentials(subscription_key)
        )

    def _convert_bounding_box(self, bounding_box: List[float]):
        left = bounding_box[0]
        top = bounding_box[1]
        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[7] - bounding_box[1]
        return top, left, width, height

    """
    OCR: Read File using the Read API, extract text - remote
    """

    def extract_document(self, read_image):
        #read_image_path = Path(file_path)
        #print(read_image_path)

        # Open the image
        #prueba = open('.\Facturas_Corregidas\Antilur\Antilur_1.png', "rb")
        #print(type(prueba))

        # Call API with image and raw response (allows you to get the operation location)
        read_response = self.azure_ocr_client.read_in_stream(read_image, raw=True)

        # Get the operation location (URL with an ID at the end) from the response
        read_operation_location = read_response.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = read_operation_location.split("/")[-1]

        # Call the "GET" API and wait for it to retrieve the results
        while True:
            read_result = self.azure_ocr_client.get_read_result(operation_id)
            if read_result.status not in ["notStarted", "running"]:
                break
            time.sleep(1)

        extraction = {
            "text": [],
            "top": [],
            "left": [],
            "width": [],
            "height": [],
            "conf": [],
        }
        # Print the detected text, line by line
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    extraction["text"].append(line.text)
                    extraction["conf"].append(line.appearance.style.confidence)
                    top, left, width, height = self._convert_bounding_box(
                        line.bounding_box
                    )
                    extraction["top"].append(top)
                    extraction["left"].append(left)
                    extraction["width"].append(width)
                    extraction["height"].append(height)

        print(extraction)
        return extraction

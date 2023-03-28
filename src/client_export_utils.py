import cv2
from pathlib import Path
import shutil
import uuid
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import simplejson as json
from constants import (
    INFERENCE_CURRENT_EXEC_CROPPED_IMAGES_PATH,
    INFERENCE_CURRENT_EXEC_JSON_PATH,
)

#Abre archivo de Google SpreadSheet
def gdrive_SpreadSheet_open(cred_json):
    print("Accediendo a Tabla Spreadsheets")
    # use creds to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(cred_json, scope)
    client = gspread.authorize(creds)
    return client



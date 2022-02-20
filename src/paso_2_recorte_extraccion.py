from numpy.lib.function_base import copy
from pyzbar.pyzbar import decode
import pytesseract as tess
from pytesseract import Output
from PIL import Image
import re
import cv2
import numpy as np
import gspread
import os
from os import path
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import easygui
import math
from scipy import ndimage
import shutil
import datetime
import pdf2image
import tkinter as tk
import simplejson as json
import copy
import funciones
import io
import time 

#Accepted file extensions to be processed
accepted_file_extension = [".jpg", ".jpeg", ".png"]

#Original Files Directory
Original_Files_Directory = ".\\Facturas_Originales"

#Processed Files Directory
Adjusted_Files_Directory = ".\\Facturas_Corregidas"

#Json Data Directory
Json_Files_Directory = ".\\Json_Facturas"

#Crop Files Directory
Crop_Files_Directory = ".\\Facturas_Recortadas"

#Templates Files Directory
Templates_Files_Directory = ".\\Facturas_Templates"

folders_to_process = ["Abasur","Antilur","La Banderita"]#,"Marioni","Masula","Los Nietitos"]


#Walk into directory looking for files to process
for root, dirs, files in os.walk(Adjusted_Files_Directory, topdown=False):
    for filename in files:
        extension = os.path.splitext(filename)[1]
        name = os.path.splitext(filename)[0]
        folder = root.split('\\')[-1]
        #Ruta al archivo Imagen del Template
        template_img_file_name = os.path.join(root.replace(Adjusted_Files_Directory, Templates_Files_Directory), folder+ "_template.png")
        #Ruta al archivo Json del Template
        template_data_file_name = os.path.join(root.replace(Adjusted_Files_Directory, Templates_Files_Directory), folder+ "_template.json")
        #Ruta al archivo Imagen Recortada
        img_recortada_file_name = os.path.join(root.replace(Adjusted_Files_Directory, Crop_Files_Directory), name+extension)
        #Ruta al archivo Json de la Imagen a procesar
        img_data_file_name = os.path.join(root.replace(Adjusted_Files_Directory, Json_Files_Directory), name+".json")

        if folder in folders_to_process and extension in accepted_file_extension and not os.path.isfile(img_recortada_file_name):
            print(" ")
            print("Processing...    "+filename)
            #Cargo Json de la imagen a  procesar
            img_data = open(img_data_file_name)
            img_data = json.load(img_data)
            #Cargo Json de template
            template_data = open(template_data_file_name)
            template_data = json.load(template_data)
            

            #Get the file to be processed
            print("Loading Image...")
            img_to_be_processed = cv2.imread(os.path.join(root, filename))
            template_to_be_processed = cv2.imread(template_img_file_name)

            #Adjust Image so text is legible
            print("Matching Template on Image...")
            img_to_be_processed,matching_result = funciones.template_match(template_to_be_processed,img_to_be_processed)
            
            if matching_result:
                #Adjust Image so text is legible
                print("Extracting OCR from Image...")
                
                is_success, im_buf_arr = cv2.imencode(".png", img_to_be_processed)
                bytes_img_to_be_processed = io.BytesIO(im_buf_arr)
                #print(type(bytes_img_to_be_processed))

                img_to_be_processed,img_data["ORC_Data_Crop"] = funciones.get_ocr_data_azure(bytes_img_to_be_processed,img_to_be_processed)

                #Extract data from image
                print("Extracting data OCR...")
                img_data = funciones.get_ocr_data_keypoints(img_to_be_processed,template_data, img_data,retry_ocr = False)

                #Save proccessed file in same location
                print("Saving Cropped Image...")
                if not os.path.exists(root.replace(Adjusted_Files_Directory, Crop_Files_Directory)):
                    os.makedirs(root.replace(Adjusted_Files_Directory, Crop_Files_Directory), exist_ok=False)
                cv2.imwrite(img_recortada_file_name, img_to_be_processed)

                #Save data file
                print("Saving Processed Image Data...")
                if not os.path.exists(root.replace(Original_Files_Directory, Json_Files_Directory)):
                    os.makedirs(root.replace(Original_Files_Directory, Json_Files_Directory), exist_ok=False)
                img_data_file = open(img_data_file_name, "w")
                img_data_file.write(json.dumps(img_data, indent=4, sort_keys=True))
                img_data_file.close()
            else:
                #Template no detectado
                print("Template no detectado en Image Data...") 
    time.sleep(2)


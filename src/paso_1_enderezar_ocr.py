import cv2
import os
from os import path
import simplejson as json
import Funciones

#Accepted file extensions to be processed
Accepted_file_extension = [".jpg", ".jpeg", ".png"]

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

folders_to_process = ["Abasur","Antilur","La Banderita","Marioni","Masula"]#,"Los Nietitos"]


#Walk into directory looking for files to process
for root, dirs, files in os.walk(".\\Facturas_Originales", topdown=False):
    for filename in files:
        extension = os.path.splitext(filename)[1]
        name = os.path.splitext(filename)[0]
        folder = root.split('\\')[-1]

        #Cambio los nombres de los archivos para que tengan nombre consecutivo Folder_01...
        if extension in Accepted_file_extension and folder not in name and folder in folders_to_process :
            counter = 1
            orig_filename = filename
            name = folder+"_"+str(counter)
            filename=folder+"_"+str(counter)+extension
            img_typo_file_name = os.path.join(root, folder+"_"+str(counter)+extension)
            while os.path.exists(img_typo_file_name):
                name = folder+"_"+str(counter)
                filename=folder+"_"+str(counter)+extension
                img_typo_file_name = os.path.join(root, folder+"_"+str(counter)+extension)
                counter = counter + 1
            img_orig_file_name = os.path.join(root, orig_filename)
            #Reemplazando Archivo Original 
            os.rename(img_orig_file_name, img_typo_file_name)
            
        #Ruta al archivo Json de Imagen Original
        img_data_file_name = os.path.join(root.replace(Original_Files_Directory, Json_Files_Directory), name+".json")
        #Ruta al archivo de la Imagen corregida
        processed_image_file_name = os.path.join(root.replace(Original_Files_Directory, Adjusted_Files_Directory), name+extension)

        #Proceso la imagen si no existe el archivo de Imagen Corregida
        if extension in Accepted_file_extension and not os.path.isfile(processed_image_file_name) and folder in folders_to_process :
            print(" ")
            print("Processing...    "+filename)
            img_data = {}
            img_data["name"] = str(filename)
            img_data["extension"] = str(extension)

            #Get the file to be processed
            print("Loading Image...")
            img_to_be_processed = cv2.imread(os.path.join(root, filename))

            #Adjust Image size to match Max Size
            print("Resizing Image...")
            img_to_be_processed,aux_img_data = Funciones.adjust_image_size(img_to_be_processed)
            img_data.update(aux_img_data)

            #Adjust Image rotation to straighten lines
            print("Rotating Image...")
            if not os.path.isfile(img_data_file_name):
                img_to_be_processed,img_data["img_correction_angle"] = Funciones.auto_adjust_image_rotation(img_to_be_processed)
            else:
                img_to_be_processed,img_data["img_correction_angle"] = Funciones.manual_adjust_image_rotation(img_to_be_processed)

            #Adjust Image so text is legible
            print("Text Orienting Image...")
            img_to_be_processed,img_data["img_orientation"] = Funciones.adjust_image_orientation(img_to_be_processed)
            
            #Add the Orientation to the correction angle
            img_data["img_correction_angle"] = img_data["img_correction_angle"] - img_data["img_orientation"]

            #Adjust Image so text is legible
            print("Extracting OCR from Image...")
            img_to_be_processed,img_data["ORC_Data"] = Funciones.get_ocr_data(img_to_be_processed)

            #Save proccessed file in same location
            print("Saving Processed Image...")
            if not os.path.exists(root.replace(Original_Files_Directory, Adjusted_Files_Directory)):
                os.makedirs(root.replace(Original_Files_Directory, Adjusted_Files_Directory), exist_ok=False)
            cv2.imwrite(processed_image_file_name, img_to_be_processed)

            #Save data file
            print("Saving Processed Image Data...")
            if not os.path.exists(root.replace(Original_Files_Directory, Json_Files_Directory)):
                os.makedirs(root.replace(Original_Files_Directory, Json_Files_Directory), exist_ok=False)
            img_data_file = open(img_data_file_name, "w")
            img_data_file.write(json.dumps(img_data, indent=4, sort_keys=True))
            img_data_file.close()


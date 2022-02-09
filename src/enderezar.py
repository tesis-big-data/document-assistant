from pyzbar.pyzbar import decode
import pytesseract as tess
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

#Accepted file extensions to be processed
accepted_file_extension=[".jpg",".jpeg",".png"]

#Function to correct image rotation and get content perfectly squared
def adjust_image_rotation(Image_To_Convert,kernel = np.ones((3,3), np.uint8)):

    img_gray = cv2.cvtColor(Image_To_Convert, cv2.COLOR_BGR2GRAY)
    img_gray= cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,41,12)
    #img_canny = cv2.erode(img_gray, kernel, iterations=3)
    #img_canny = cv2.dilate(img_canny, kernel, iterations=1)

    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    #cv2.imshow("Ventana",img_edges)
    #cv2.waitKey()
    try:
        lines = cv2.HoughLinesP(img_edges,rho = 1,theta = 1*math.pi/180,threshold = 200,minLineLength = 100,maxLineGap = 50)
        
        lines_touple = []
        angle_corrections = []

        for lines_det in lines:
            x1, y1, x2, y2 = lines_det[0]

            length= ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)
            angle = math.degrees(math.atan2(x2 - x1, y1 - y2))

            lines_touple.append((angle,length,x1, y1, x2, y2))
            #ret=cv2.line(Image_lines, (x1,y1), (x2,y2), (0, 255, 0), 2)
           
        #data = np.array([i[0] for i in lines_touple])
        #plt.hist(data, bins=90)
        #plt.show()
        lines_touple.sort(key = lambda x: x[1])

        if len(lines_touple)>20:
            lines_touple = lines_touple[-20:]
        
        for angle,length,x1, y1, x2, y2 in lines_touple:
            cv2.line(Image_To_Convert, (x1,y1), (x2,y2), (255, 0, 255), 2)
            distancia_0 = np.abs(angle-0)
            distancia_90 = np.abs(angle-90)
            distancia_180 = np.abs(angle-180)

            if distancia_0 <= distancia_90 and distancia_0 <= distancia_180:
                angle_corrections.append(angle)
            elif distancia_90 <= distancia_0 and distancia_90 <= distancia_180:
                angle_corrections.append(angle-90)
            elif distancia_180 <= distancia_0 and distancia_180 <= distancia_90:
                angle_corrections.append(angle-180)

        correction_angle = sum(angle_corrections)/len(angle_corrections)

        Image_To_Convert = ndimage.rotate(Image_To_Convert,correction_angle,cval=255)

    except:
        print("No se encuentran lineas de alineación se saltea archivo")


    return  Image_To_Convert

#Function to correct text orientation in image, ensures text is allways correct
def adjust_image_orientation(Image_To_Convert):
    newdata=tess.image_to_osd(Image_To_Convert)
    Image_To_Convert = ndimage.rotate(Image_To_Convert,-int(re.search('(?<=Rotate: )\d+', newdata).group(0)),cval=255)
    return  Image_To_Convert

#Function to crop image and meet a specific size 
def adjust_image_size(Image_To_Convert, Max_Size = 1700):
    #Get height and width of the image
    img_height=Image_To_Convert.shape[0]
    img_width=Image_To_Convert.shape[1]

    #Get the bigger size to resize to match Max_Size
    scale_percent = 0
    if img_width>img_height:
        scale_percent = Max_Size*100/img_width # Percent of original size from width
    else:
        scale_percent = Max_Size*100/img_height # Percent of original size from height

    #Get the dimensions of the new image
    width = int(img_width * scale_percent / 100)
    height = int(img_height * scale_percent / 100)
    dim = (width, height)

    #Resize image
    if scale_percent > 0:
        Image_To_Convert = cv2.resize(Image_To_Convert, dim, interpolation = cv2.INTER_AREA)
        return  Image_To_Convert

    return  False


#Walk into directory looking for files to process
for root, dirs, files in os.walk(".", topdown=False):
   for filename in files:
       extension = os.path.splitext(filename)[1]
       name = os.path.splitext(filename)[0]
       if extension in accepted_file_extension:
            print (" ")
            print ("Processing...    "+filename)
            #Get the file to be processed
            img_to_be_processed = cv2.imread(os.path.join(root, filename))

            #Adjust Image size to match Max Size
            img_to_be_processed = adjust_image_size(img_to_be_processed)

            #Adjust Image rotation to straighten lines
            img_to_be_processed = adjust_image_rotation(img_to_be_processed)

            #Adjust Image so text is legible
            img_to_be_processed = adjust_image_orientation(img_to_be_processed)

            #Save proccessed file in same location
            nombre_archivo=os.path.join(root, name+"_ADJ"+extension)
            cv2.imwrite(nombre_archivo,img_to_be_processed)

   for name in dirs:
      print(os.path.join(root, name))


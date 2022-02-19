import cv2
import numpy as np
import math
from numpy.lib.function_base import copy
import pytesseract as tess
from pytesseract import Output
import re
import cv2
import numpy as np
from scipy import ndimage
import copy

#Función que toma template_img lo busca en orig_image y retorna el template encontrado en orig_image y el resultado del ajuste.
#Distancia es el umbral para aceptar el ajuste del template, menor distancia mas estricto el ajuste
#Min_match_count es la cantidad minima de keypoints similares para ser considerado un match
def template_match(template_img,orig_image,distance=0.7,min_match_count=4):
    try:
        #Preparo las imagenes a comparar
        gray1 = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

        #Creo el objeto SIFT
        sift = cv2.xfeatures2d.SIFT_create()

        #Creo flann matcher
        matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})

        #Detecto keypoints y computo los descriptores
        kpts1, descs1 = sift.detectAndCompute(gray1,None)
        kpts2, descs2 = sift.detectAndCompute(gray2,None)

        #Realizo knnMatch para obtener los dos mejores matches
        matches = matcher.knnMatch(descs1, descs2, 2)
        #Ordeno los matches según distancia
        matches = sorted(matches, key = lambda x:x[0].distance)

        #Testeo ratios para tenere buenos matches
        good = [m1 for (m1, m2) in matches if m1.distance < distance * m2.distance]


        if len(good)>min_match_count:
            src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        else:
            print( "No se encontraron suficientes matches - {}/{}".format(len(good),min_match_count))
            return orig_image,False
            

        #Recorto el template encontrado de la imagen original
        h,w = template_img.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
        found_image = cv2.warpPerspective(orig_image,perspectiveM,(w,h))

        #Retorno template encontrado de la imagen original
        return found_image,True
    except:
        return orig_image,False

#Función para buscar los datos del template en la factura a analizar, 
#devuelve diccionario con los valores detectados
def get_ocr_data_keypoints(Image_To_Convert, json_template_data, json_img_data):
    for keypoint in json_template_data:
        if keypoint in json_img_data:
            del json_img_data[keypoint]
        if not (keypoint == "extension" or keypoint == "name"):
            for i in range(len(json_img_data["ORC_Data_Crop"]["text"])):
                x_centro_palabra = json_img_data["ORC_Data_Crop"]["left"][i] + round(json_img_data["ORC_Data_Crop"]["width"][i]/2)
                y_centro_palabra = json_img_data["ORC_Data_Crop"]["top"][i] + round(json_img_data["ORC_Data_Crop"]["height"][i]/2)
                x_inicial_template=json_template_data[keypoint][0][0]
                y_inicial_template=json_template_data[keypoint][0][1]
                x_final_template=json_template_data[keypoint][1][0]
                y_final_template=json_template_data[keypoint][1][1]
                if x_inicial_template <= x_centro_palabra <= x_final_template:
                    if y_inicial_template <= y_centro_palabra <= y_final_template:
                        if keypoint in json_img_data:
                            json_img_data[keypoint]= json_img_data[keypoint] + " " + json_img_data["ORC_Data_Crop"]["text"][i]
                        else:
                            json_img_data[keypoint]=json_img_data["ORC_Data_Crop"]["text"][i]

            if keypoint not in json_img_data:
                img_aux,crop_ocr = get_ocr_data(Image_To_Convert[y_inicial_template:y_final_template, x_inicial_template:x_final_template])
                #cv2.imshow("crop",img_aux)
                if len(crop_ocr["text"])>0:
                    for word in crop_ocr["text"]:
                        if keypoint in json_img_data:
                                json_img_data[keypoint]= json_img_data[keypoint] + " " + word
                        else:
                                json_img_data[keypoint]="(CROP) " + word
    return json_img_data

#Función para corregir manualmente la rotación de la imagen
#Devuelve Imagen rotada a cuadrante mas cercano, y angulo de rotación
def manual_adjust_image_rotation(Image_To_Convert):
    def on_change(value):
        adaptiveThreshold_1 = cv2.getTrackbarPos('adaptiveThreshold_1', "Rotacion")
        adaptiveThreshold_2 = cv2.getTrackbarPos('adaptiveThreshold_2', "Rotacion")
        cannyEdges_1 = cv2.getTrackbarPos('cannyEdges_1', "Rotacion")
        cannyEdges_2 = cv2.getTrackbarPos('cannyEdges_2', "Rotacion")
        linesThreshold_1 = cv2.getTrackbarPos('linesThreshold_1', "Rotacion")
        linesThreshold_2 = cv2.getTrackbarPos('linesThreshold_2', "Rotacion")
        print(str(adaptiveThreshold_1)+" "+str(adaptiveThreshold_2)+" "+str(cannyEdges_1)+" "+str(cannyEdges_2))

        img_gray = cv2.cvtColor(Image_To_Convert, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, adaptiveThreshold_1, adaptiveThreshold_2)
        grey_3_channel = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_edges = cv2.Canny(img_gray, cannyEdges_1, cannyEdges_2, L2gradient = True)
        lines = cv2.HoughLinesP(img_edges, rho=1, theta=1*math.pi / 180, threshold=linesThreshold_1, minLineLength=10, maxLineGap=linesThreshold_2)
        finalImage = copy.deepcopy(Image_To_Convert)
        if lines is not None:
            print("----------------")
            for lines_det in lines:
                print(lines_det)
                x1, y1, x2, y2 = lines_det[0]
                cv2.line(finalImage, (x1,y1), (x2,y2), (0, 255, 255), 5)

        imS = np.hstack((cv2.resize(grey_3_channel, (round(grey_3_channel.shape[1]*600/grey_3_channel.shape[0]), 600)) , cv2.resize(finalImage, (round(finalImage.shape[1]*600/finalImage.shape[0]), 600)) ))
        cv2.imshow("Rotacion", imS)

    try:
        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    except AttributeError:
        flags = cv2.WINDOW_NORMAL
    cv2.namedWindow("Rotacion",flags)    # Create window with freedom of dimensions
    cv2.createTrackbar('adaptiveThreshold_1', "Rotacion", 31, 250, on_change)
    cv2.createTrackbar('adaptiveThreshold_2', "Rotacion", 20, 250, on_change)
    cv2.createTrackbar('cannyEdges_1', "Rotacion", 100, 500, on_change)
    cv2.createTrackbar('cannyEdges_2', "Rotacion", 200, 500, on_change)
    cv2.createTrackbar('linesThreshold_1', "Rotacion", 200, 600, on_change)
    cv2.createTrackbar('linesThreshold_2', "Rotacion", 100, 600, on_change)
    
    imS = cv2.resize(Image_To_Convert, (round(Image_To_Convert.shape[1]*600/Image_To_Convert.shape[0]), 600))  # Resize image

    adaptiveThreshold_1 = cv2.getTrackbarPos('adaptiveThreshold_1', "Rotacion")
    adaptiveThreshold_2 = cv2.getTrackbarPos('adaptiveThreshold_2', "Rotacion")
    cannyEdges_1 = cv2.getTrackbarPos('cannyEdges_1', "Rotacion")
    cannyEdges_2 = cv2.getTrackbarPos('cannyEdges_2', "Rotacion")
    linesThreshold_1 = cv2.getTrackbarPos('linesThreshold_1', "Rotacion")
    linesThreshold_2 = cv2.getTrackbarPos('linesThreshold_2', "Rotacion")

    correction_angle = 0
    img_gray = cv2.cvtColor(Image_To_Convert, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, adaptiveThreshold_1, adaptiveThreshold_2)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    #img_gray = cv2.erode(img_gray, kernel)
    #img_gray = cv2.dilate(img_gray, kernel)
    grey_3_channel = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    img_edges = cv2.Canny(img_gray, cannyEdges_1, cannyEdges_2, L2gradient = True)
    
    try:
        lines = cv2.HoughLinesP(img_edges, rho=1, theta=1*math.pi /
                                180, threshold=linesThreshold_1, minLineLength=10, maxLineGap=linesThreshold_2)

        lines_touple = []
        angle_corrections = []
        print("--------LINES--------")
        for lines_det in lines:
            #print(lines_det)
            x1, y1, x2, y2 = lines_det[0]

            length = ((((x2 - x1)**2) + ((y2-y1)**2))**0.5)
            angle = math.degrees(math.atan2(x2 - x1, y1 - y2))
            #print(angle)

            lines_touple.append((angle, length, x1, y1, x2, y2))
    
        if len(lines_touple) == 0:
            print("No lines No adjustment")
            return Image_To_Convert,correction_angle  # No lines No adjustment

        lines_touple.sort(key=lambda x: x[1])

        #if len(lines_touple) > 20:
        #    lines_touple = lines_touple[-20:]

        for angle, length, x1, y1, x2, y2 in lines_touple:
            distancia_0 = np.abs(angle-0)
            distancia_90 = np.abs(angle-90)
            distancia_180 = np.abs(angle-180)

            if distancia_0 <= distancia_90 and distancia_0 <= distancia_180:
                angle_corrections.append(angle)
            elif distancia_90 <= distancia_0 and distancia_90 <= distancia_180:
                angle_corrections.append(angle-90)
            elif distancia_180 <= distancia_0 and distancia_180 <= distancia_90:
                angle_corrections.append(angle-180)
        print("-----Angles-----")
        #print(angle_corrections)
        print('Mean:', np.mean(angle_corrections))
        print('Standard Deviation:', np.std(angle_corrections))
        angle_corrections = [x for x in angle_corrections if (x >= (np.mean(angle_corrections)-np.std(angle_corrections)) and x <= (np.mean(angle_corrections)+np.std(angle_corrections)))]
        #plt.hist(angle_corrections,bins='auto', density=True)
        #plt.show()
        correction_angle = sum(angle_corrections)/len(angle_corrections)
        #json_data["img_correction_angle"] = correction_angle
        print("Angle... " + str(correction_angle))

        Image_To_Convert = ndimage.rotate(
            Image_To_Convert, correction_angle, cval=255)

    except:
        print("No se encuentran lineas de alineación se saltea archivo")

    return Image_To_Convert,correction_angle

#Función para corregir manualmente la rotación de una imagen
#Devuelve Imagen rotada a cuadrante mas cercano, y angulo de rotación
def auto_adjust_image_rotation(Image_To_Convert):
    adaptiveThreshold_1 = 31
    adaptiveThreshold_2 = 20
    correction_angle=0

    img_gray = cv2.cvtColor(Image_To_Convert, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, adaptiveThreshold_1, adaptiveThreshold_2)

    img_edges = cv2.Canny(img_gray, 100, 200, L2gradient = True)

    try:
        lines = cv2.HoughLinesP(img_edges, rho=1, theta=1*math.pi /
                                180, threshold=200, minLineLength=10, maxLineGap=100)

        lines_touple = []
        angle_corrections = []

        for lines_det in lines:
            x1, y1, x2, y2 = lines_det[0]

            length = ((((x2 - x1)**2) + ((y2-y1)**2))**0.5)
            angle = math.degrees(math.atan2(x2 - x1, y1 - y2))

            lines_touple.append((angle, length, x1, y1, x2, y2))
        
        if len(lines_touple) == 0:
            return Image_To_Convert,correction_angle  # No lines No adjustment

        lines_touple.sort(key=lambda x: x[1])

        #if len(lines_touple) > 20:
        #    lines_touple = lines_touple[-20:]

        for angle, length, x1, y1, x2, y2 in lines_touple:
            distancia_0 = np.abs(angle-0)
            distancia_90 = np.abs(angle-90)
            distancia_180 = np.abs(angle-180)

            if distancia_0 <= distancia_90 and distancia_0 <= distancia_180:
                angle_corrections.append(angle)
            elif distancia_90 <= distancia_0 and distancia_90 <= distancia_180:
                angle_corrections.append(angle-90)
            elif distancia_180 <= distancia_0 and distancia_180 <= distancia_90:
                angle_corrections.append(angle-180)
        print("-----Angles-----")
        #print(angle_corrections)
        print('Mean:', np.mean(angle_corrections))
        print('Standard Deviation:', np.std(angle_corrections))
        angle_corrections = [x for x in angle_corrections if (x >= (np.mean(angle_corrections)-np.std(angle_corrections)) and x <= (np.mean(angle_corrections)+np.std(angle_corrections)))]
        #plt.hist(angle_corrections,bins='auto', density=True)
        #plt.show()
        correction_angle = sum(angle_corrections)/len(angle_corrections)
        #json_data["img_correction_angle"] = correction_angle
        print("Angle... " + str(correction_angle))

        Image_To_Convert = ndimage.rotate(Image_To_Convert, correction_angle, cval=255)

    except:
        print("No se encuentran lineas de alineación se saltea archivo")

    return Image_To_Convert,correction_angle

#Función para corregir orientación de la pagina para que texto quede bien orientado
#Devuelve Imagen rotada a cuadrante mas cercano, y angulo de rotación
def adjust_image_orientation(Image_To_Convert):
    OCR_Orientation_Angle = 0
    try:
        #Get OCR Orientation Data
        newdata = tess.image_to_osd(Image_To_Convert)
        #Get Orientation Angle
        OCR_Orientation_Angle = int(
            re.search('(?<=Rotate: )\d+', newdata).group(0))
    except:
        OCR_Orientation_Angle = 90

    #Rotate only if needed
    if OCR_Orientation_Angle != 0:
        Image_To_Convert = ndimage.rotate(
            Image_To_Convert, -OCR_Orientation_Angle, cval=255)

    return Image_To_Convert,OCR_Orientation_Angle

#Función para extraer el OCR de una imagen, devuelve imagen procesada y datos de OCR
def get_ocr_data(Image_To_Convert, min_conf = 0, deleted_words=[""," "], draw_rectangles=False):
    #Get OCR Data from Image
    imagedataocr = tess.image_to_data(
        Image_To_Convert, output_type=Output.DICT)

    #Generate OCR Data Copy
    imagedataocrcopy = copy.deepcopy(imagedataocr)

    #Get OCR Data Length
    n_boxes = len(imagedataocr['conf'])
    #Auxiliary Index for Deleting
    i_adj = 0

    #Verify all OCR Data and delete Confidences = -1 or deleted words
    for i in range(n_boxes):
        if int(imagedataocr['conf'][i]) > min_conf and (imagedataocr['text'][i] not in deleted_words):
            (x, y, w, h) = (imagedataocr['left'][i], imagedataocr['top']
                            [i], imagedataocr['width'][i], imagedataocr['height'][i])
            if draw_rectangles:
                cv2.rectangle(Image_To_Convert, (x, y),(x + w, y + h), (0, 255, 0), 2)
        else:
            imagedataocrcopy['block_num'].pop(i-i_adj)
            imagedataocrcopy['conf'].pop(i-i_adj)
            imagedataocrcopy['height'].pop(i-i_adj)
            imagedataocrcopy['left'].pop(i-i_adj)
            imagedataocrcopy['level'].pop(i-i_adj)
            imagedataocrcopy['line_num'].pop(i-i_adj)
            imagedataocrcopy['page_num'].pop(i-i_adj)
            imagedataocrcopy['par_num'].pop(i-i_adj)
            imagedataocrcopy['text'].pop(i-i_adj)
            imagedataocrcopy['top'].pop(i-i_adj)
            imagedataocrcopy['width'].pop(i-i_adj)
            imagedataocrcopy['word_num'].pop(i-i_adj)
            i_adj = i_adj + 1

    return Image_To_Convert,imagedataocrcopy

#Función para extraer el OCR de una imagen con servicio de Azure, devuelve imagen procesada y datos de OCR
def get_ocr_data_azure(Image_To_Convert, min_conf = 0, deleted_words=[""," "], draw_rectangles=False):
    #Get OCR Data from Image
    imagedataocr = tess.image_to_data(
        Image_To_Convert, output_type=Output.DICT)

    #Generate OCR Data Copy
    imagedataocrcopy = copy.deepcopy(imagedataocr)

    #Get OCR Data Length
    n_boxes = len(imagedataocr['conf'])
    #Auxiliary Index for Deleting
    i_adj = 0

    #Verify all OCR Data and delete Confidences = -1 or deleted words
    for i in range(n_boxes):
        if int(imagedataocr['conf'][i]) > min_conf and (imagedataocr['text'][i] not in deleted_words):
            (x, y, w, h) = (imagedataocr['left'][i], imagedataocr['top']
                            [i], imagedataocr['width'][i], imagedataocr['height'][i])
            if draw_rectangles:
                cv2.rectangle(Image_To_Convert, (x, y),(x + w, y + h), (0, 255, 0), 2)
        else:
            imagedataocrcopy['conf'].pop(i-i_adj)
            imagedataocrcopy['height'].pop(i-i_adj)
            imagedataocrcopy['left'].pop(i-i_adj)
            imagedataocrcopy['text'].pop(i-i_adj)
            imagedataocrcopy['top'].pop(i-i_adj)
            imagedataocrcopy['width'].pop(i-i_adj)
            i_adj = i_adj + 1

    return Image_To_Convert,imagedataocrcopy

#Función para ajustar el tamaño de la imagen, devuelve imagen con tamaño ajustado
def adjust_image_size(Image_To_Convert, Max_Size=1700):
    #Get height and width of the image
    img_height = Image_To_Convert.shape[0]
    img_width = Image_To_Convert.shape[1]

    #Get the bigger size to resize to match Max_Size
    scale_percent = 0
    if img_width > img_height:
        scale_percent = Max_Size*100/img_width  # Percent of original size from width
    else:
        scale_percent = Max_Size*100/img_height  # Percent of original size from height

    #Get the dimensions of the new image
    width = int(img_width * scale_percent / 100)
    height = int(img_height * scale_percent / 100)
    dim = (width, height)

    #Resize image
    if scale_percent > 0:
        Image_To_Convert = cv2.resize(
            Image_To_Convert, dim, interpolation=cv2.INTER_AREA)
    json_data = {}
    json_data["orig_height"] = img_height
    json_data["orig_width"] = img_width
    json_data["scale_percent"] = scale_percent
    json_data["adj_height"] = height
    json_data["adj_width"] = width

    return Image_To_Convert, json_data



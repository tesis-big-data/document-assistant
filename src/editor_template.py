import cv2
import os
from scipy import ndimage
import simplejson as json

# Accepted file extensions to be processed
accepted_file_extension = [".jpg", ".jpeg", ".png"]

# Original Files Directory
Original_Files_Directory = ".\\Facturas_Originales"

# Processed Files Directory
Adjusted_Files_Directory = ".\\Facturas_Corregidas"

# Json Data Directory
Json_Files_Directory = ".\\Json_Facturas"

# Crop Files Directory
Crop_Files_Directory = ".\\Facturas_Recortadas"

# Templates Files Directory
Templates_Files_Directory = ".\\Facturas_Templates"

folders_to_process = [
    "Abasur",
    "Antilur",
    "La Banderita",
    "Marioni",
    "Masula",
    "Los Nietitos",
]

# Lists to store the bounding box coordinates
top_left_corner = []
bottom_right_corner = []
# Dict for bounding boxes
modo_dict = {
    0: {"txt_modo": "Giro", "color_modo": (0, 0, 0)},
    1: {"txt_modo": "Recorte", "color_modo": (0, 255, 0)},
    2: {"txt_modo": "Numero", "color_modo": (255, 0, 0)},
    3: {"txt_modo": "Fecha", "color_modo": (0, 0, 255)},
    4: {"txt_modo": "Total", "color_modo": (255, 255, 0)},
    5: {"txt_modo": "Cliente", "color_modo": (0, 255, 255)},
}
# General Variables
drawing = False
angulo = 0.0
modo = 0
original_image = None
factor = 1
resize_image = None
resize_image_editable = None
modificaciones = {}

# Function which will be called on mouse input
def drawRectangle(action, x, y, flags, *userdata):
    # Referencing global variables
    global top_left_corner, bottom_right_corner, drawing, modo, modo_dict, resize_image_editable, resize_image

    if modo == 0:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1
    lineType = 2

    # Mark the top left corner when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        top_left_corner = [(x, y)]
        drawing = True
    # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP:
        resize_image_editable = resize_image.copy()
        bottom_right_corner = [(x, y)]
        drawing = False
        # Draw the rectangle
        cv2.rectangle(
            resize_image_editable,
            top_left_corner[0],
            bottom_right_corner[0],
            modo_dict[modo]["color_modo"],
            1,
            8,
        )
        if modo > 0:
            cv2.putText(
                resize_image_editable,
                modo_dict[modo]["txt_modo"],
                top_left_corner[0],
                font,
                fontScale,
                modo_dict[modo]["color_modo"],
                thickness,
                lineType,
            )
        cv2.imshow("Window", resize_image_editable)

    elif action == cv2.EVENT_MOUSEMOVE and drawing:
        bottom_right_corner = [(x, y)]
        resize_image_editable = resize_image.copy()
        cv2.rectangle(
            resize_image_editable,
            top_left_corner[0],
            bottom_right_corner[0],
            modo_dict[modo]["color_modo"],
            1,
            8,
        )
        if modo > 0:
            cv2.putText(
                resize_image_editable,
                modo_dict[modo]["txt_modo"],
                top_left_corner[0],
                font,
                fontScale,
                modo_dict[modo]["color_modo"],
                thickness,
                lineType,
            )
        cv2.imshow("Window", resize_image_editable)


# Walk into directory looking for files to process
for root, dirs, files in os.walk(Templates_Files_Directory, topdown=False):
    for filename in files:
        extension = os.path.splitext(filename)[1]
        name = os.path.splitext(filename)[0]
        folder = root.split("\\")[-1]

        template_name = os.path.join(root, folder + "_template" + extension)
        template_name_campos = os.path.join(root, folder + "_campos" + extension)
        template_data_file_name = os.path.join(root, folder + "_template" + ".json")

        if (
            folder in folders_to_process
            and extension in accepted_file_extension
            and not os.path.isfile(template_data_file_name)
        ):
            # Create a named window
            cv2.namedWindow("Window")
            # highgui function called when mouse events occur
            cv2.setMouseCallback("Window", drawRectangle)
            modificaciones = {}
            angulo = 0.0
            modo = 0

            print(" ")
            print("Template...    " + template_name)
            img_data = {}
            img_data["name"] = str(template_name)
            img_data["extension"] = str(extension)

            # Get the file to be processed
            print("Loading Image...")
            original_image = cv2.imread(os.path.join(root, name + extension))
            if original_image.shape[1] > original_image.shape[0]:
                factor = (
                    700 / original_image.shape[1]
                )  # Percent of original size from width
            else:
                factor = 700 / original_image.shape[0]
            resize_image = cv2.resize(original_image, None, fx=factor, fy=factor)
            resize_image_editable = resize_image.copy()

            k = 0
            while k != 27:
                # Display the image
                cv2.imshow("Window", resize_image_editable)
                top_left_corner = []
                bottom_right_corner = []
                k = cv2.waitKey(0)
                if modo == 0 and (k == 97):
                    angulo = angulo + 0.1
                    resize_image_editable = ndimage.rotate(
                        resize_image, angulo, cval=255
                    )
                    cv2.imshow("Window", resize_image_editable)
                elif modo == 0 and (k == 100):
                    angulo = angulo - 0.1
                    resize_image_editable = ndimage.rotate(
                        resize_image, angulo, cval=255
                    )
                    cv2.imshow("Window", resize_image_editable)
                elif modo == 0 and (k == 115):
                    angulo = angulo + 10
                    resize_image_editable = ndimage.rotate(
                        resize_image, angulo, cval=255
                    )
                    cv2.imshow("Window", resize_image_editable)
                elif modo == 0 and (k == 119):
                    angulo = angulo - 10
                    resize_image_editable = ndimage.rotate(
                        resize_image, angulo, cval=255
                    )
                    cv2.imshow("Window", resize_image_editable)
                elif k == 32:
                    if modo == 0:
                        original_image = ndimage.rotate(
                            original_image, angulo, cval=255
                        )
                        resize_image = cv2.resize(
                            original_image, None, fx=factor, fy=factor
                        )
                        resize_image_editable = resize_image
                        modo = 1
                    elif modo >= 1:
                        if len(top_left_corner) > 0 and len(bottom_right_corner) > 0:
                            if modo == 1:
                                original_image = original_image[
                                    int(1 / factor * top_left_corner[0][1]) : int(
                                        1 / factor * bottom_right_corner[0][1]
                                    ),
                                    int(1 / factor * top_left_corner[0][0]) : int(
                                        1 / factor * bottom_right_corner[0][0]
                                    ),
                                ]
                                resize_image = cv2.resize(
                                    original_image, None, fx=factor, fy=factor
                                )
                                resize_image_editable = resize_image
                            else:
                                modificaciones[modo_dict[modo]["txt_modo"]] = [
                                    (
                                        int(1 / factor * top_left_corner[0][0]),
                                        int(1 / factor * top_left_corner[0][1]),
                                    ),
                                    (
                                        int(1 / factor * bottom_right_corner[0][0]),
                                        int(1 / factor * bottom_right_corner[0][1]),
                                    ),
                                ]
                                resize_image = resize_image_editable

                    cv2.imshow("Window", resize_image)
                elif k == 49:
                    modo = 1
                elif k == 50:
                    modo = 2
                elif k == 51:
                    modo = 3
                elif k == 52:
                    modo = 4
                elif k == 53:
                    modo = 5
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 1
            lineType = 2
            resize_image_editable = original_image
            for key in modificaciones:
                cv2.rectangle(
                    resize_image_editable,
                    modificaciones[key][0],
                    modificaciones[key][1],
                    (255, 0, 0),
                    1,
                    8,
                )
                img_data[key] = [modificaciones[key][0], modificaciones[key][1]]
                cv2.putText(
                    resize_image_editable,
                    key,
                    modificaciones[key][0],
                    font,
                    fontScale,
                    (255, 0, 0),
                    thickness,
                    lineType,
                )

            # Save data file
            print("Saving Processed Image Data...")
            img_data_file = open(template_data_file_name, "w")
            img_data_file.write(json.dumps(img_data, indent=4, sort_keys=True))
            img_data_file.close()

            cv2.imwrite(template_name, original_image)
            cv2.imwrite(template_name_campos, resize_image_editable)
            cv2.destroyAllWindows()


########################git branch -M main
### IMPORT PACKAGES
########################

import cv2
import os
import math

import numpy as np
import pandas as pd
#import tensorflow as tf
#import layoutparser as lp #Something more modern from PaddleOCR

from typing import Tuple, Union
from deskew import determine_skew
from PIL import Image
from pdf2image import convert_from_path
from paddleocr import PPStructure, draw_structure_result, save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

########################
### SETUP 
########################

#PDF path
pdf_path = './sample_data/'

#PDF name
pdf_name = '2008PDFTest_AssmtRpt05'

#Extension
pdf_ext = '.pdf'

#Output path - output path generated upon image upload?
out_path = './sample_results/' + pdf_name + '/'

########################
### IMAGE PROCESSING FUNCTIONS 
########################

def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2

    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def normalize(image: np.ndarray, norm_min: int, norm_max: int) -> np.ndarray:

    #Image may already be normalized
    if np.min(image) != norm_min or np.max(image) != norm_max:
        
        #Output image array
        norm_img = np.zeros((image.shape[0], image.shape[1]))

        #Normalize based on min/max        
        image_norm = cv2.normalize(image, norm_img, norm_min, norm_max, cv2.NORM_MINMAX)

    else:
        image_norm = image

    #Return normalized image
    return image_norm

def scale(image: np.ndarray, resolution: float) -> np.ndarray:

    #Convert to pillow library
    PIL_image = Image.fromarray(image)

    length_x, width_y = PIL_image.size

    #Scale image to desired maximum image dimension
    factor = max(1, float(resolution / length_x))
    size = int(factor * length_x), int(factor * width_y)

    #Resize using PIL
    im_resized = PIL_image.resize(size, Image.Resampling.LANCZOS)

    #Return numpy array
    return np.array(im_resized)

########################
### TABLE RECOVERY 
########################

#Paddle Paddle Table Engine
#table_engine = PPStructure(show_log=True, lang='en') #Image orientation didin't work i.e. image_orientation=True, missing paddleclas
table_engine = PPStructure(recovery=True, lang='en')

########################
### Convert PDF to images
########################

#Convert PDF into images indexed on PDF page
images = convert_from_path(pdf_path + pdf_name + pdf_ext)

########################
### LOOP
########################

#List of exel tables
excel_tables = []

#Save converted images to directory
for i in range(len(images)):
    
    #Save the image for processing comparison
    images[i].save(out_path + 'page'+str(i)+'.jpg', "JPEG")

    #Load the saved image
    image = cv2.imread(out_path + 'page'+str(i)+'.jpg')

    #Normalizaiton
    image = normalize(image, 0, 255) #Passed through object

    #Convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Greyscale

    #Determine skew angle - show to recover skews down to 2 deg in tests
    angle = determine_skew(grayscale)

    #Rotate against skew angle
    image = rotate(grayscale, angle, (0, 0, 0))

    #Image scaling
    image = scale(image, 1080) #Passed through object

    #Noise Supression
    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    #Binarization
    _, image = cv2.threshold(image, (127 + int(np.mean(image)))/2, 255, cv2.THRESH_BINARY) #Class param 255

    #Save processed image
    cv2.imwrite(out_path + 'page'+str(i)+'_processed.jpg', image)

    #Run image through table extraction algorithm
    result = table_engine(image)

    #Generalized OCR
    #Implement later

    #Table Extraction - to .xlsx
    save_structure_res(result, out_path, os.path.basename(out_path + 'page'+str(i)+'.jpg').split('.')[0])

    #Layout Analysis
    font_path = './PaddleOCR/doc/fonts/simfang.ttf' # English
    image = Image.open(out_path + 'page'+str(i)+'.jpg').convert('RGB')
    im_show = draw_structure_result(image, result,font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save(out_path + 'page'+str(i)+'/layout.jpg')
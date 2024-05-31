# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:43:56 2021

@author: david
"""

#%reset -f

import numpy as np
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk

#***************************************************
#* Funciones ***************************************
#***************************************************
def normaliza(a):
    a=a.astype(np.double)
    a=a/a.max()*255
    b=a.astype(np.uint8)
    return b


def cmykar(img):

    scale = 255
    percent = 0.5    

    # separate b,g,r

    b,g,r = cv2.split(img)
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)

    # convert to cmyk
    # see 
    # https://stackoverflow.com/questions/14088375/how-can-i-convert-rgb-to-cmyk-and-vice-versa-in-python/41220097
    # https://www.codeproject.com/Articles/4488/XCmyk-CMYK-to-RGB-Calculator-with-source-code
    
    c = 1 - r / scale
    m = 1 - g / scale
    y = 1 - b / scale
    k = cv2.min(cv2.min(c, m),y)
    c = scale * (c - k) / (1 - k)
    m = scale * (m - k) / (1 - k)
    y = scale * (y - k) / (1 - k)
       
    
    # desaturate neighbors of G which are C,Y
    # c = cv2.multiply(c, percent)
    # y = cv2.multiply(y, percent)
 
    c=normaliza(c)
    m=normaliza(m)
    y=normaliza(y)
    k=normaliza(k)
    
    return c,m,y,k

def imgshow(img,label='Imagen'):
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 



#  2 - Determina los valores máximos y mínimos sin ceros

def rango(a):
    minval_a = np.min(a[np.nonzero(a)])
    maxval_a = np.max(a[np.nonzero(a)])
    return minval_a, maxval_a

#  3 - Binariza dejando en cero los valores fuera del rango min max

def binariza(a,minval,maxval):
    a[a<minval]=0;
    a[a>maxval]=0;
    a[a>0]=255;
    return a

#  4 - Rutina cambia tamaño de la imagen en porcentaje

def cambia_tamano(img,a):
    escala = a # debe ser decimal o veces
    w = int(img.shape[1] * escala)
    h = int(img.shape[0] * escala)
    dim = (w, h)
    img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

#  5 - Rutina cambia tamaño de la imagen en ancho y largo

def cambia_tamano_wh(img,w,h):
    dim = (w, h)
    img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

#  6 - Rutina de muestra con texto y espera cualquier tecla

def muestra(txt,img,w=640,h=420):
    img=cambia_tamano_wh(img,w,h)
    cv2.imshow(txt,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return

#  7 - Rutina de muestra con texto y espera escape

def muestra_esc(txt,img):
    img=cambia_tamano_wh(img,640,420)
    cv2.imshow(txt,img)
    while True:
        if  cv2.waitKey(10)==27:
            break
    cv2.destroyAllWindows()
    return

#  8 - Rutina de muestra con texto y espera escape un segundo

def muestra_1s(txt,img):
    img=cambia_tamano_wh(img,640,420)
    cv2.imshow(txt,img)
    cv2.waitKey(120)
    cv2.destroyAllWindows() 
    return

#  9 - Rutina crea waffer de una imagen de una capa

def bw2waffer(img):
    img=np.dstack((img,img,img))
    return img


# 10 - Rutina para dibijar datos

def dibuja(xlabel, ylabel, data):
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return


def detect_text_lines(blurred_image):
    # Obtener las dimensiones de la imagen
    height, width = blurred_image.shape
    
    # Inicializar una lista para almacenar las líneas de texto
    text_lines = []
    
    # Umbral para considerar una línea como texto (puede ajustarse según sea necesario)
    threshold = 2000
    
    # Barrido horizontal
    for y in range(height):
        row = blurred_image[y, :]
        count_white = np.sum(row == 255)
        
        if count_white > threshold:
            text_lines.append(y)
    
    # Agrupar líneas adyacentes para formar renglones
    grouped_lines = []
    current_group = []
    for i in range(len(text_lines) - 1):
        if text_lines[i + 1] - text_lines[i] <= 1:
            current_group.append(text_lines[i])
        else:
            if current_group:
                grouped_lines.append(current_group)
                current_group = []
    if current_group:
        grouped_lines.append(current_group)

def draw_text_lines(original_image, text_regions):
    for y_start, y_end in text_regions:
        cv2.rectangle(original_image, (0, y_start), (original_image.shape[1], y_end), (0, 255, 0), 2)

    return original_image

def detect_characters_in_line(binary_image, y_start, y_end):
    line_image = binary_image[y_start:y_end, :]
    contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    return char_bounding_boxes

def hit_or_miss(character_roi, templates):
    # Comparar la ROI del carácter con cada plantilla y devolver la mejor coincidencia
    best_match = None
    best_score = float('inf')
    
    for char, template in templates.items():
        # Redimensionar la plantilla a las dimensiones de la ROI
        resized_template = cv2.resize(template, (character_roi.shape[1], character_roi.shape[0]))
        score = np.sum(np.abs(character_roi - resized_template))
        
        if score < best_score:
            best_score = score
            best_match = char
    
    return best_match



def dilate(image):
    kernel=np.ones(kernel, np.uint8)
    erode=image.copy()
    dilate = cv2.dilate(erode,kernel)
    for i in range(0, 4):
        pass
        #dilate = cv2.dilate(erode, kernel)
        #a=("Erosionado{"+str(i+1)+"} times")
        #imgshow(dilate,a)

    return erode


def preprocessing(img_path,filter=False,kernel = (3, 3)):
        # ****** Escala de Grises **************
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if filter == 'gaussian': # ****** Filtro Gaussiano **************
        imagen_filtered = cv2.GaussianBlur(image, kernel, 2) #Suavizado de la imagen: "Desenfoque"
    else:
        imagen_filtered = image

    # Binarizar la imagen (Umbral en 127 para convertir la imagen a binaria)
    _, binary_image = cv2.threshold(imagen_filtered, 127, 255, cv2.THRESH_BINARY)

    

    return binary_image

def process_image(image, templates):
    # Cargar y preprocesar la imagen
    blurred_image = preprocessing(image)
    
    # Binarizar la imagen
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)
    
    # Detectar líneas de texto
    text_regions = detect_text_lines(binary_image)
    
    # Dibujar las regiones de texto en la imagen original
    processed_image = draw_text_lines(image, text_regions)
    
    for y_start, y_end in text_regions:
        char_bounding_boxes = detect_characters_in_line(binary_image, y_start, y_end)
        
        for x, y, w, h in char_bounding_boxes:
            roi = binary_image[y_start + y:y_start + y + h, x:x + w]
            char = hit_or_miss(roi, templates)
            cv2.rectangle(processed_image, (x, y_start + y), (x + w, y_start + y + h), (255, 0, 0), 2)
            cv2.putText(processed_image, char, (x, y_start + y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Convertir la imagen de BGR a RGB para Matplotlib
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # Mostrar la imagen con Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(processed_image_rgb)
    plt.title('Detección de Cadenas de Caracteres y Caracteres Individuales')
    plt.axis('off')
    plt.show()

















# Función para seleccionar una imagen
def select_image():
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    root.destroy()
    return file_path

from skimage.feature import hog

# Función de comparación con plantillas
def match_template(char_img, templates):
    best_match = None
    min_diff = float('inf')
    
    for char, template in templates.items():
        resized_img = cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_AREA)
        diff = np.sum((resized_img - template) ** 2)
        
        if diff < min_diff:
            min_diff = diff
            best_match = char
            
    return best_match

def create_character_template(char, font=cv2.FONT_HERSHEY_SIMPLEX, size=1, thickness=2):
    # Crear una imagen en blanco
    img = np.zeros((28, 28), dtype=np.uint8)
    
    # Obtener el tamaño del texto
    text_size = cv2.getTextSize(char, font, size, thickness)[0]
    
    # Posición del texto centrada
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    
    # Poner el texto en la imagen
    cv2.putText(img, char, (text_x, text_y), font, size, (255), thickness, lineType=cv2.LINE_AA)
    
    return img

def setTemplates(debug=False):
    # Crear plantillas para números y signos específicos
    characters = '0123456789.$,ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    templates = {char: create_character_template(char) for char in characters}
    if debug:
        plt.figure(figsize=(10, 2))
        for i, (char, template) in enumerate(templates.items()):
            plt.subplot(1, len(templates), i + 1)
            plt.imshow(template, cmap='gray')
            plt.title(char)
            plt.axis('off')
        plt.show()
    return templates

def extract_features(character_img):
    character_img_resized = cv2.resize(character_img, (28, 28), interpolation=cv2.INTER_AREA)
    features = hog(character_img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    return features

def segment(binary_image):
    # Encontrar contornos en la imagen binarizada
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = binary_image[y:y+h, x:x+w]
        rois.append((roi, x, y, w, h))
    
    return rois

def segment_characters(roi):
    # Aplicar umbral adaptativo para binarizar la imagen de la ROI
    binary_roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Encuentra los contornos de los caracteres
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar los contornos de izquierda a derecha
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Extraer los caracteres
    characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_img = binary_roi[y:y+h, x:x+w]
        characters.append(char_img)
    
    return characters

def drawRois(original_image):
    # Convertir la imagen de BGR a RGB para Matplotlib
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Mostrar la imagen con Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image_rgb)
    plt.title('ROIs en la Imagen Original')
    plt.axis('off')
    plt.show()


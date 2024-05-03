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
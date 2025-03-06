# -*- coding: utf-8 -*-
"""
Segmentación con Canny
Created on Wed Mar  5 21:30:36 2025

@author: juanc
"""

import cv2
import numpy as np

img = cv2.imread('iphone.jpg')

GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(GrayImg, 25, 150)
#Vamos a dilatar, convolucion
kernel = np.ones((5,5),np.uint8)
#Vamos ampliar los bordes
bordesdilatados = cv2.dilate(canny, kernel)

contours,_ = cv2.findContours(bordesdilatados, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
objetos = bordesdilatados.copy()
#Con esta funcion vamos a obtener la mascara del elemento, entero de 8bits
cv2.drawContours(objetos, [max(contours, key=cv2.contourArea)], -1, 255,thickness =-1) 

#Vamos a realizar la segmentacion usando la mascara
#Lo que este en 0 queda en 0 y lo que este en 255 se vuelve 1
objetos = objetos / 255 

segmentacion = np.zeros(img.shape)
##Dejar 0 todo lo que no sea del celular y en 1 lo que si sea el celular
#Lo que hacemos es dejar el fondo gris al sumarle 125*(objetos==0) donde el objeto sea 0
segmentacion[:,:,0] = objetos *  img[:,:,0] + 125*(objetos==0)
segmentacion[:,:,1] = objetos *  img[:,:,1] + 125*(objetos==0)
segmentacion[:,:,2] = objetos *  img[:,:,2] + 125*(objetos==0)
#Como esta en un flotamte de 64bits lo vamos a pasar a un entero de 8 bits
segmentacion = np.uint8(segmentacion)
cv2.imshow('original', img)
cv2.imshow('imagen segmentada', segmentacion)
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 20:45:08 2025

@author: juanc
"""

import cv2
import numpy as np


a = np.zeros((50,100))
b = np.ones((50,100))

#img = np.uint8(255*np.concatenate((a,b), axis=0))  

img = cv2.imread('bananos.jpg',0)

gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, 5)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, 5)


magnitud, angulo = cv2.cartToPolar(gx, gy)

#Por temas de visualizacion (Muchos valores de blanco) obtenemos el valor 
#absoluto(negativo a Positivo ) y escalar la imagen
#Los valores maximos 255 y los valores sean menores de 255
gx = cv2.convertScaleAbs(gx)
gy = cv2.convertScaleAbs(gy)
magnitud = cv2.convertScaleAbs(magnitud)

angulo = (180/np.pi)*angulo #Pasamos a grados por lo que lo da en radianes
#LaPlaceano usando un filtro gausssian
imgFilter = cv2.GaussianBlur(img, (5,5), 0)
laplaceano = cv2.convertScaleAbs(cv2.Laplacian(imgFilter, cv2.CV_64F,5))
#Algoritmo que realza los bordes, en los valores donde es 255 es el borde
# el resto puede ser ruido de la imagen
canny = cv2.Canny(img, 25,150)
cv2.imshow('Canny',canny )
#cv2.imshow('gx',gx)
#cv2.imshow('gy',gy)
#cv2.imshow('magnitud',magnitud)
#cv2.imshow('Laplaceano',laplaceano)
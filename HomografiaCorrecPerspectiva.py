# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 20:40:53 2025

@author: juanc
"""
import cv2
import numpy as np


img = cv2.imread('placa.jpg')

imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

filas,columnas,canales = img.shape;

#Cuadro
#pts_iniciales = np.array([[42,12],[193,51],[40,303],[198,268]])
#Placa
pts_iniciales = np.array([[465,138],[1339,350],[518,706],[1399,981]])

pts_destino = np.array([[0,0],[columnas,0],[0,filas],[columnas,filas]])

h,_ = cv2.findHomography(pts_iniciales, pts_destino)

im2 = cv2.warpPerspective(img, h, (columnas,filas))

cv2.imshow('original', img)
cv2.imshow('Fix perpectiva', im2)
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:13:03 2025

@author: juanc
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
 
img = cv2.imread('000078.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

umbral,_ = cv2.threshold(I, 0, 255, cv2.THRESH_OTSU)

mascara = np.uint8((I<umbral)*255)

output = cv2.connectedComponentsWithStats(mascara,4,cv2.CV_32S)
cantObt = output[0]
labels = output[1]
stats = output[2]

mascara = (np.argmax(stats[:,4][1:])+1==labels)
mascara = ndimage.binary_fill_holes(mascara).astype(int)
mascara1 = np.uint8(mascara*255)
contours,_ = cv2.findContours(mascara1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

P = cv2.arcLength(cnt, True)
A = cv2.contourArea(cnt)
A1 = np.sum(mascara1/255)

#CONVEX HULL consiste en el poligono convexo mas pequeño con puntos relacionados
hull = cv2.convexHull(cnt)
puntosConvex = hull[:,0,:]
m,n = mascara1.shape
ar = np.zeros((m,n))
mascaConvex = np.uint8(cv2.fillConvexPoly(ar,puntosConvex,1))


#BoundingBox, la caja que muestra el objeto de interes roatado
#Computar el cuadrado con area minima
recmin = cv2.minAreaRect(cnt)
box = np.int0(cv2.boxPoints(recmin))
m,n = mascara1.shape
ar = np.zeros((m,n))
mascaraRect = np.uint8(cv2.fillConvexPoly(ar, box, 1))

#BoundingBox, la caja que muestra el objeto de interes recto
x,y, ancho,alto = cv2.boundingRect(cnt)
cv2.rectangle(img, (x,y), (x+ancho,y+alto),(0,255,0),1)

#Muestra el hull convex
contours,_ = cv2.findContours(mascaConvex, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,0,255),1)
#Muestra el elemento en su orientacion
contours,_ = cv2.findContours(mascaraRect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (255,0,0),1)


dato = I.flatten()

rojo = img[:,:,0].flatten()
verde = img[:,:,1].flatten()
azul = img[:,:,2].flatten()

#plt.hist(rojo,bins=1000,histtype='stepfilled',color="red")
#plt.hist(verde,bins=1000,histtype='stepfilled',color="green")
#plt.hist(azul,bins=1000,histtype='stepfilled',color="blue")


segmentadaColor = np.zeros((m,n,3)).astype('uint8')
segmentadaColor[:,:,0] = np.uint8(img[:,:,0]*mascara) 
segmentadaColor[:,:,1] = np.uint8(img[:,:,1]*mascara)
segmentadaColor[:,:,2] = np.uint8(img[:,:,2]*mascara)

segmentaGrey = np.zeros((m,n))
segmentaGrey[:,:] = np.uint8(I*mascara)
cv2.imshow('imagen', segmentadaColor)
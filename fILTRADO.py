# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 19:48:33 2025

@author: juanc
"""

import cv2
import numpy as np


img = cv2.imread('bananos.jpg')

cv2.imshow('original', img)

# kernel3x3 = np.ones((3,3),np.float32) / (3*3)
# output = cv2.filter2D(img, -1, kernel3x3)
# cv2.imshow('promedio 3x3', output)

# kernel3x3 = np.ones((5,5),np.float32) / (5*5)
# output = cv2.filter2D(img, -1, kernel3x3)
# cv2.imshow('promedio 5x5', output)

# kernel3x3 = np.ones((31,31),np.float32) / (31*31)
# output = cv2.filter2D(img, -1, kernel3x3)
# cv2.imshow('promedio 31x31', output)

output = cv2.GaussianBlur(img, (3,3), 0)
cv2.imshow('gauss sigma o desviacion 3*3', output)

output = cv2.GaussianBlur(img, (11,11), 0)
cv2.imshow('gauss sigma o desviacion 11*11', output)

output = cv2.GaussianBlur(img, (21,21), 0)
cv2.imshow('gauss sigma o desviacion 21*21', output)


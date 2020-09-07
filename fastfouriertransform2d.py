# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:06:31 2020

@author: muadgra

Fast fourier transform in 2d domain and applying low-pass filter to it.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

plt.figure(figsize=(20, 4.8*5), constrained_layout=False)

img = cv2.imread("ben.jpg", 0)

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

original = np.fft.fft2(img)
center = np.fft.fftshift(original)

plt.subplot(151), plt.imshow(img, "gray"), plt.title("Original Image")

#plt.subplot(151), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Spectrum")

LowPass = idealFilterLP(50,img.shape)
plt.subplot(152), plt.imshow(np.abs(LowPass), "gray"), plt.title("Low Pass Filter")

LowPassCenter = center * idealFilterLP(25,img.shape)
plt.subplot(153), plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray"), plt.title("Centered Spectrum multiply Low Pass Filter")

LowPass = np.fft.ifftshift(LowPassCenter)
plt.subplot(154), plt.imshow(np.log(1+np.abs(LowPass)), "gray"), plt.title("Decentralize")

inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(155), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Processed Image")
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 00:01:55 2016

@author: zlzhong
"""
# new commit

import numpy as np
import scipy
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io

mat_gt = scipy.io.loadmat('/home/finoa/DL-on-HSI-Classification/Datasets/UPavia/PaviaU_gt.mat')
gt = mat_gt['paviaU_gt']

mat_data = scipy.io.loadmat('/home/finoa/DL-on-HSI-Classification/Datasets/UPavia/PaviaU.mat')
data = mat_data['paviaU']


x = np.ravel(gt)
#print x
y = np.zeros((x.shape[0], 3))

for index, item in enumerate(x):
    if item == 0:
        y[index] = np.array([0,0,0])/255.
    if item == 1:
        y[index] = np.array([128,128,128])/255.
    if item == 2:
        y[index] = np.array([128,0,0])/255.
    if item == 3:
        y[index] = np.array([128,0,128])/255.
    if item == 4:
        y[index] = np.array([0,255,255])/255.
    if item == 5:
        y[index] = np.array([0,255,255])/255.
    if item == 6:
        y[index] = np.array([255,0,255])/255.
    if item == 7:
        y[index] = np.array([255,0,0])/255.
    if item == 8:
        y[index] = np.array([255,255,0])/255.
    if item == 9:
        y[index] = np.array([0,128,0])/255.
        
        
#print y

plt.figure(1)
y_re = np.reshape(y,(gt.shape[0],gt.shape[1],3))
print y_re
plt.imshow(y_re)
plt.show()


plt.figure(2)
data = data/8000.
blue = data[:, :, 10]                                                   #blue band
green = data[:, :, 24]                                                  #green band
red = data[:, :, 44]                                                    #red band
rgb_hsi = np.zeros((data.shape[0], data.shape[1], 3))
rgb_hsi[:, :, 0] = red
rgb_hsi[:, :, 1] = green
rgb_hsi[:, :, 2] = blue
plt.imshow(rgb_hsi)
plt.show()

# plt.savefig('output_pavia_gt.jpg')

#scipy.misc.imsave('outfile.jpg', y_re)
#im = Image.fromarray(y_re, 'RGB') 
#im.save('my.tif')
#im.show()

# im1 = Image.open('output_pavia_gt.jpg')
# im1_arr = np.array(im1)
# print im1_arr
# print im1_arr.shape

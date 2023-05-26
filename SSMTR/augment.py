# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:35:58 2021

@author: HLB
"""
import cv2
import numpy as np
import random


class CenterResizeCrop(object):
    '''
    Class that performs CenterResizeCrop. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    scale_begin: min scale size
    windowsize: max scale size

    -------------------------------------------------------------------------------------
    '''
    def __init__(self, scale_begin = 23, windowsize = 27):
        self.scale_begin = scale_begin
        self.windowsize = windowsize
    
    def __call__(self, image):
        
        length = np.array(range(self.scale_begin, self.windowsize+1, 2))
 
        row_center = int((self.windowsize-1)/2)
        col_center = int((self.windowsize-1)/2)   
        row = image.shape[1]
        col = image.shape[2]
        # band = image.shape[0]
        s = np.random.choice(length, size = 1)
        halfsize_row = int((s-1)/2)
        halfsize_col = int((s-1)/2)             
        r_image = image[:, row_center-halfsize_row : row_center+halfsize_row+1, col_center-halfsize_col : col_center+halfsize_col+1]        
        r_image = np.transpose(cv2.resize(np.transpose(r_image,[1,2,0]), (row, col)), [2,0,1]) 
        return r_image  
        
class RandomResizeCrop(object):
    def __init__(self, scale = [0.5, 1], probability = 0.5):
        self.scale = scale
        self.probability = probability
    
    def __call__(self, image):
        
        if random.uniform(0, 1) > self.probability:
            return image
        else:
            row = image.shape[1]
            col = image.shape[2]
            s = np.random.uniform(self.scale[0], self.scale[1])
            r_row = round(row * s)
            r_col = round(col * s)
            halfsize_row = int((r_row-1)/2)
            halfsize_col = int((r_col-1)/2)
            row_center =random.randint(halfsize_row, r_row - halfsize_row-1)
            col_center =random.randint(halfsize_col, r_col - halfsize_col-1)             
            r_image = image[:, row_center-halfsize_row : row_center+halfsize_row+1, col_center-halfsize_col : col_center+halfsize_col+1]            
            r_image = np.transpose(cv2.resize(np.transpose(r_image,[1,2,0]), (row, col)), [2,0,1]) 
            return r_image 

class CenterCrop(object):
    '''
    Class that performs CenterResizeCrop. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    scale_begin: min scale size
    windowsize: max scale size

    -------------------------------------------------------------------------------------
    '''
    def __init__(self, scale_begin = 23, windowsize = 27, probability = 0.5):
        self.scale_begin = scale_begin
        self.windowsize = windowsize
        self.probability = probability
    
    def __call__(self, image):
        
        if random.uniform(0, 1) > self.probability:
            return image
        else:
            length = np.array(range(self.scale_begin, self.windowsize, 2))
     
            row_center = int((self.windowsize-1)/2)
            col_center = int((self.windowsize-1)/2)   
            s = np.random.choice(length, size = 1)
            halfsize_row = int((s-1)/2)
            halfsize_col = int((s-1)/2)             
            r_image = image[:, row_center-halfsize_row : row_center+halfsize_row+1, col_center-halfsize_col : col_center+halfsize_col+1]
            
            # r_image = np.pad(r_image, ((0, 0), (row_center - halfsize_row, row_center - halfsize_row), (col_center - halfsize_col, col_center - halfsize_col)), 'edge')
            r_image = np.pad(r_image, ((0, 0), (row_center - halfsize_row, row_center - halfsize_row), (col_center - halfsize_col, col_center - halfsize_col)), 'constant', constant_values=0)
            return r_image     
     
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.shape[1]
        w = img.shape[2]
        c = img.shape[0]
        
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        
        mask = np.tile(mask[np.newaxis,:,:], (c,1,1))
        img = img * mask

        return img
    
def resize(train_image, size = (27,27)):
    r_image = np.zeros([train_image.shape[0], train_image.shape[1], size[0], size[1]], dtype = np.float32)
    for i in range(train_image.shape[0]):
        r_image[i] = np.transpose(cv2.resize(np.transpose(train_image[i],[1,2,0]), size), [2,0,1]) 
    return r_image
    
def take_elements(image, location, windowsize):
    if windowsize == 1:
        if len(image.shape) == 3:
            spectral = np.zeros([location.shape[0],image.shape[2]], dtype = image.dtype)
            for i in range(location.shape[0]):
                spectral[i] = image[location[i][0], location[i][1]]
        else:
            spectral = np.zeros(location.shape[0], dtype = np.int32)
            for i in range(location.shape[0]):
                spectral[i] = image[location[i][0], location[i][1]]            
    else:
        if len(image.shape) == 3:
            halfsize = int((windowsize - 1)/2)
            spectral = np.zeros([location.shape[0], windowsize, windowsize, image.shape[2]], dtype = image.dtype)
            for i in range(location.shape[0]):
                spectral[i,:,:,:] = image[location[i][0]-halfsize : location[i][0]+halfsize+1 , location[i][1]-halfsize : location[i][1]+halfsize+1, :]  
        else:
            halfsize = int((windowsize - 1)/2)
            spectral = np.zeros([location.shape[0], windowsize, windowsize], dtype = image.dtype)
            for i in range(location.shape[0]):
                spectral[i,:,:] = image[location[i][0]-halfsize : location[i][0]+halfsize+1 , location[i][1]-halfsize : location[i][1]+halfsize+1]              
    return spectral
  
    

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:26:52 2019

@author: HLB
"""

import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from build_EMP import build_emp 

def pca_whitening(image, number_of_pc):

    shape = image.shape
    
    image = np.reshape(image, [shape[0]*shape[1], shape[2]])
    number_of_rows = shape[0]
    number_of_columns = shape[1]
    pca = PCA(n_components = number_of_pc)
    image = pca.fit_transform(image)
    pc_images = np.zeros(shape=(number_of_rows, number_of_columns, number_of_pc),dtype=np.float32)
    for i in range(number_of_pc):
        pc_images[:, :, i] = np.reshape(image[:, i], (number_of_rows, number_of_columns))
    
    return pc_images

def load_data(dataset):
    if dataset == 'Indian':
        image_file = r'.\datasets/Indian\indian_pines_corrected.mat'
        label_file = r'.\datasets/Indian\Indian_pines_gt.mat'
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)
        image = image_data['indian_pines_corrected']
        label = label_data['indian_pines_gt']
    elif dataset == 'Pavia':
        image_file = r'.\datasets\Pavia\Pavia.mat'
        label_file = r'.\datasets\Pavia\Pavia_groundtruth.mat'
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)        
        image = image_data['paviaU']#pavia1
        label = label_data['paviaU_gt']#pavia1
    elif dataset == 'CASI':
        image_file = r'.\datasets\Houston\CASI.mat'
        label_file = r'.\datasets\Houston\CASI_gnd_flag.mat'
        image_data = sio.loadmat(image_file)
        label_data = sio.loadmat(label_file)        
        image = image_data['CASI']
        label = label_data['gnd_flag']  # houston 
    else:
        raise Exception('dataset does not find')
    image = image.astype(np.float32)
 
    return image, label
 

def readdata(type, dataset, windowsize, train_num, val_num, num):

    or_image, or_label = load_data(dataset)
    # image = np.expand_dims(image, 2)
    halfsize = int((windowsize-1)/2)
    number_class = np.max(or_label)
 
    image = np.pad(or_image, ((halfsize, halfsize), (halfsize, halfsize), (0, 0)), 'edge')
    label = np.pad(or_label, ((halfsize, halfsize), (halfsize, halfsize)), 'constant',constant_values=0)
               
    if type == 'PCA':
        image1 = pca_whitening(image, number_of_pc = 3)
    elif type == 'EMP':
        image1 = pca_whitening(image, number_of_pc = 4)
        num_openings_closings = 3
        emp_image = build_emp(base_image=image1, num_openings_closings=num_openings_closings)
        image1 = emp_image
    elif type == 'none':
        image1 = np.copy(image)
    else:
        raise Exception('type does not find')
    image = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1)) 
    #set the manner of selecting training samples 
        
    
    n = np.zeros(number_class,dtype=np.int64)
    for i in range(number_class):
        temprow, tempcol = np.where(label == i + 1)
        n[i] = len(temprow)    
    total_num = np.sum(n)
    
    nTrain_perClass = np.ones(number_class,dtype=np.int64) * train_num
    for i in range(number_class):
        if n[i] <=  nTrain_perClass[i]: 
            nTrain_perClass[i] = 15  
    ###
    nValidation_perClass =  (n/total_num)*val_num
    nvalid_perClass = nValidation_perClass.astype(np.int32)   
       
    index = []
    flag = 0
    fl = 0

    
    bands = np.size(image,2)    
    validation_image = np.zeros([np.sum(nvalid_perClass), windowsize, windowsize, bands], dtype=np.float32)
    validation_label = np.zeros(np.sum(nvalid_perClass), dtype=np.int64)
    train_image = np.zeros([np.sum(nTrain_perClass), windowsize, windowsize, bands], dtype=np.float32)
    train_label = np.zeros(np.sum(nTrain_perClass),dtype=np.int64)
    train_index = np.zeros([np.sum(nTrain_perClass), 2], dtype = np.int32)              
    val_index =  np.zeros([np.sum(nvalid_perClass), 2], dtype = np.int32)   
       
    for i in range(number_class):        
        temprow, tempcol = np.where(label == i + 1)
        matrix = np.zeros([len(temprow),2], dtype=np.int64)
        matrix[:,0] = temprow
        matrix[:,1] = tempcol
        np.random.seed(num)
        np.random.shuffle(matrix)
        
        temprow = matrix[:,0]
        tempcol = matrix[:,1]         
        index.append(matrix)

        for j in range(nTrain_perClass[i]):
            train_image[flag + j, :, :, :] = image[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                            (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            train_label[flag + j] = i
            train_index[flag + j] = matrix[j,:]
        flag = flag + nTrain_perClass[i]

        for j in range(nTrain_perClass[i], nTrain_perClass[i] + nvalid_perClass[i]):
            validation_image[fl + j-nTrain_perClass[i], :, :,:] = image[(temprow[j] - halfsize):(temprow[j] + halfsize + 1),
                                                   (tempcol[j] - halfsize):(tempcol[j] + halfsize + 1)]
            validation_label[fl + j-nTrain_perClass[i] ] = i 
            val_index[fl + j-nTrain_perClass[i]] = matrix[j,:]
        fl =fl + nvalid_perClass[i]
        

    return train_image, train_label, validation_image, validation_label,nTrain_perClass, nvalid_perClass,train_index, val_index, index, image, label,total_num

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:21:37 2022

@author: malkhatib
"""
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
from operator import truediv
import random 
from sklearn.utils import shuffle
from numpy.fft import fft2, fftshift


def loadData(name):
    data_path = os.path.join(os.getcwd(),'data')

    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'PC':
        data = sio.loadmat(os.path.join(data_path, 'Pavia.mat'))['pavia']
        labels = sio.loadmat(os.path.join(data_path, 'Pavia_gt.mat'))['pavia_gt']
    elif name == 'BO':
        data = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
    elif name == 'GP':
        data = sio.loadmat(os.path.join(data_path, 'Gulfport.mat'))['gulfport']
        labels = sio.loadmat(os.path.join(data_path, 'Gulfport_gt.mat'))['gulfport_gt']
    elif name == 'WHU_Hi_LK':
        data = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
        labels = sio.loadmat(os.path.join(data_path, 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
    return data, labels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def target(name):
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                        ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']
    elif name == 'BO':
        target_names = ['Water','Hippo_grass','Floodpain_grasses_1','Floodpain_grasses_2','Reeds','Riparian','Firescar',
                        'Island_interior','Acacia_woodlands','Acacia_shrublands','Acacia_grasslands','Short_mopane',
                        'Mixed_mopane','Exposed_soils']
    elif name == 'GP':
        target_names = ['Tree', 'Shadow', 'Grass', 'Dead Grass', 'Asphalt', 'Dirt']
        
    return target_names 
    
def num_classes(dataset):
    if dataset == 'PU' or dataset == 'PC':
        output_units = 9
    elif dataset == 'IP' or dataset == 'SA':
        output_units = 16
    elif dataset == 'BO':
        output_units = 14
    elif dataset == 'GP':
        output_units = 6
    elif dataset == 'WHU_Hi_LK':
        output_units = 9

    return output_units




def Patch(data,height_index,width_index, PATCH_SIZE):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch

def getTrainTestSplit(X, y, pxls_num):
    if type(pxls_num) != list:
        pxls_num = [pxls_num]*len(np.unique(y))
        
    if len(np.unique(y)) != len(pxls_num):
        print("length of pixels list doen't match the number of classes in the dataset")
        return
    else:
        xTrain = []
        yTrain = []
        xTest  = []
        yTest  = []
        for i in range(len(np.unique(y))):
            if pxls_num[i] > len(y[y==i]):
                print("Number of training pixles is larger than total class pixels")
                return
            else:
                random.seed(123) #optional to reproduce the data
                samples = random.sample(range(len(y[y==i])), pxls_num[i])
                xTrain.extend(X[y==i][samples])
                yTrain.extend(y[y==i][samples])
                
                tmp1 = list(X[y==i])
                tmp2 = list(y[y==i])
                for ele in sorted(samples, reverse = True):
                    del tmp1[ele]
                    del tmp2[ele]
                xTest.extend(tmp1)
                yTest.extend(tmp2)
     
  
    xTrain, yTrain = shuffle(xTrain, yTrain, random_state=321)  
    xTest, yTest = shuffle(xTest, yTest, random_state=345) 
    
    return np.array(xTrain), np.array(xTest), np.array(yTrain) , np.array(yTest)
        
        
    
def getFFT(X):
    X_fft = np.zeros(X.shape, dtype='complex64')
    for ii in range(len(X)):
        for jj in range(X.shape[3]):
            X_fft[ii,:,:,jj] = fftshift(fft2(X[ii,:,:,jj])) 
            #X_fft[ii,:,:,jj] = fftshift(fft2(X[ii,:,:,jj])) 
            
            
    return X_fft

















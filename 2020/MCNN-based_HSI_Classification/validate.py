import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization, Lambda, AveragePooling2D, Add
from keras.layers import Dropout, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

from operator import truediv
import time
from subpixel_conv2d import SubpixelConv2D
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import argparse

# MODELS = ['MCNN-CP','Oct-MCNN-PS'] # 可选模型
# DATASETS = ['IP','UH','UP','SA'] # 可选数据集
# RATIOS = [0.99, 0.995] # 可选测试样本量(测试样本+验证样本)，剩余为训练样本量  IP:0.99  UH/UP/SA:0.995
parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--dataset', '-d', type=str, default='IP',
                    help='dataset name')
parser.add_argument('--model', '-m', type=str, default='Oct-MCNN-PS',
                    help='model name')
parser.add_argument('--ratio', '-r', default=0.99, type=float,
                    help='test ratio')
args = parser.parse_args()

def loadData(name):
    data_path = os.path.join(os.getcwd(),'data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'UH':
        data = sio.loadmat(os.path.join(data_path, 'HoustonU.mat'))['houstonU'] # 601*2384*50
        labels = sio.loadmat(os.path.join(data_path, 'HoustonU_gt.mat'))['houstonU_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'UP':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    return data, labels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,stratify=y)
    return X_train, X_test, y_train, y_test

def applyPCA(X, numComponents=64):
    newX = np.reshape(X, (-1, X.shape[2]))
    print(newX.shape)
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca, pca.explained_variance_ratio_

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createPatches(X, y, windowSize=25, removeZeroLabels = True):
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

def infoChange(X,numComponents):
    X_copy = np.zeros((X.shape[0] , X.shape[1], X.shape[2]))
    half = int(numComponents/2)
    for i in range(0,half-1):
        X_copy[:,:,i] = X[:,:,(half-i)*2-1]
    for i in range(half,numComponents):
        X_copy[:,:,i] = X[:,:,(i-half)*2]
    X = X_copy
    return X

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports (X_test,y_test,name):
    start = time.time()
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    end = time.time()
    print(end - start)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                        'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'UH':
        target_names = ['Healthy grass','Stressed grass','Artificial turf','Evergreen trees', 'Deciduous trees','Bare earth','Water',
                        'Residential buildings','Non-residential buildings','Roads','Sidewalks','Crosswalks','Major thoroughfares','Highways',
                       'Railways','Paved parking lots','Unpaved parking lots','Cars','Trains','Stadium seats']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'UP':
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']
    
    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32) 
    Test_Loss =  score[0]*100
    Test_accuracy = score[1]*100
    
    return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100


def main():
    ## GLOBAL VARIABLES
    dataset = args.dataset
    modelname = args.model
    test_ratio = args.ratio
    train_ratio = int((1-test_ratio)*100)

    windowSize = 11
    componentsNum = 20 if dataset == 'UP' else 28
    if dataset == 'UP':
      componentsNum = 15
    elif dataset == 'UH':
      componentsNum = 50 if test_ratio >= 0.99 else 25
    elif dataset == 'IP':
      componentsNum = 110
    else:
      componentsNum = 30
    drop = 0.4
    
    X, y = loadData(dataset)
    X,pca,ratio = applyPCA(X,numComponents=componentsNum)
    X = infoChange(X,componentsNum)
    X, y = createPatches(X, y, windowSize=windowSize)
    
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)
    Xvalid, Xtest, yvalid, ytest = splitTrainTestSet(Xtest, ytest, (test_ratio-train_ratio/train_val_ratio)/test_ratio)

    Xtest = Xtest.reshape(-1, windowSize, windowSize, componentsNum, 1)
    ytest = np_utils.to_categorical(ytest)

    model = load_model('models/'+str(modelname)+'.h5', custom_objects={'SubpixelConv2D': SubpixelConv2D,'tf': tf})
    adam = Adam(lr=0.001, decay=1e-06)

    model.load_weights('pretrained_models/'+str(modelname)+"_"+str(dataset)+".hdf5")
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    Y_pred_test = model.predict(Xtest)
    y_pred_test = np.argmax(Y_pred_test, axis=1)
    classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
    print(classification)

    classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtest,ytest,dataset)
    classification = str(classification)
    confusion1 = str(confusion)
    file_name = "classification_report.txt"

    with open(file_name, 'w') as x_file:
        x_file.write('{} Test loss (%)'.format(Test_loss))
        x_file.write('\n')
        x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion1))

if __name__ == '__main__':
    main()


from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt

import scipy
from utils import *
from model_DualPath_realAndFFT import dual_Path, dual_Path_with_SE


## GLOBAL VARIABLES
dataset = 'PU' 
train_percentage = 0.01 
test_ratio = 1 - train_percentage
windowSize = 13 
PCA_comp = 15 

X, y = loadData(dataset)


# Apply PCA for dimensionality reduction
X,pca = applyPCA(X,PCA_comp)

X, y = createImageCubes(X, y, windowSize=windowSize)


Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(np.concatenate([X,getFFT(X)], axis = 3), y, test_ratio)


# Split Real and FFT components
X_train = Xtrain[:,:,:,:Xtrain.shape[3]//2].real
X_fft_train = Xtrain[:,:,:,Xtrain.shape[3]//2:]

X_test = Xtest[:,:,:,:Xtrain.shape[3]//2].real
X_fft_test = Xtest[:,:,:,Xtrain.shape[3]//2:]



ytrain = keras.utils.to_categorical(ytrain)
ytest = keras.utils.to_categorical(ytest)
X_train = np.expand_dims(X_train, axis=4)
X_test = np.expand_dims(X_test, axis=4)
X_fft_train = np.expand_dims(X_fft_train, axis=4)
X_fft_test = np.expand_dims(X_fft_test, axis=4)


    ###############################################################################
# Callbacks
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='loss', 
                              patience=10,
                              restore_best_weights=True
                              )




model = dual_Path_with_SE(X_train, X_fft_train, num_classes(dataset))
    
model.summary()


    
history = model.fit({"x":X_train,"x_fft":X_fft_train}, ytrain,
                        batch_size = 16, 
                        verbose=1, 
                        epochs=100, 
                        shuffle=True, 
                        #class_weight = class_weights,
                        callbacks = [early_stopper])
    
    
Y_pred_test = model.predict([X_test, X_fft_test] )
y_pred_test = np.argmax(Y_pred_test, axis=1)
  
    
kappa = cohen_kappa_score(np.argmax(ytest, axis=1),  y_pred_test)
oa = accuracy_score(np.argmax(ytest, axis=1), y_pred_test)
confusion = confusion_matrix(np.argmax(ytest, axis=1), y_pred_test)
each_acc, aa = AA_andEachClassAccuracy(confusion)
       
    
print("OA = ", oa) 
print("AA = ", aa)
print('Kappa = ', kappa)
  




# load the original image
X, y = loadData(dataset)
height = y.shape[0]
width = y.shape[1]
X,pca = applyPCA(X, numComponents=PCA_comp)
X = padWithZeros(X, windowSize//2)
model = dual_Path_with_SE(X_train, X_fft_train, num_classes(dataset))
model.summary()

# Generate the predicted image, a pixel/patch wise operation, will take long time
outputs = np.zeros((height,width))
for i in range(height):
    for j in range(width):
        target = int(y[i,j])
        if i%25 == 0 and j%25 ==0: 
            print("i = " + str(i) + ", j = " + str(j))
        if target == 0 :
            continue
        else :
            image_patch = Patch(X,i,j, windowSize)
            tmp = np.expand_dims(image_patch, axis=0)
            image_patch_FFT = np.expand_dims(getFFT(tmp), axis = 4)
            image_patch_real = image_patch.reshape(1, image_patch.shape[0],
                                               image_patch.shape[1], 
                                               image_patch.shape[2], 1).astype('float64')  




                                 
            prediction = model.predict([image_patch_real,image_patch_FFT], verbose = 0)
            prediction = np.argmax(prediction, axis=1)
            outputs[i][j] = prediction+1
 

scipy.io.savemat('./SE_Dual_' + dataset +'.mat', {'outputs': outputs})

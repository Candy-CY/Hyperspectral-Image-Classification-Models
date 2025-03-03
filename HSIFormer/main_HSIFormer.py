
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
import scipy
from utils import *
from model_HSIFormer import HSIFormer


## GLOBAL VARIABLES
dataset = 'PU' #input("select the data set:\nPU for pavia\nSA for Salinas\nIP for indian pines\nGP for GulfPort\n") 
train_percentage = 0.01 #float(input("Enter the Train percentage\n"))/100
test_ratio = 1 - train_percentage
windowSize = 15 #int(input("Enter the window size\n"))
PCA_comp = 10 #int(input("Enter the number of PCs\n"))

X1, y = loadData(dataset)


# Apply PCA for dimensionality reduction
X2,pca = applyPCA(X1,PCA_comp)


X, yy = createImageCubes(X2, y, windowSize=windowSize)

Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, yy, test_ratio)



#ytrain =keras.utils.to_categorical(ytrain)
#ytest =keras.utils.to_categorical(ytest)
Xtrain = np.expand_dims(Xtrain, axis=4)
Xtest = np.expand_dims(Xtest, axis=4)
X = np.expand_dims(X, axis=4)


model = HSIFormer(input_shape=(windowSize, windowSize, PCA_comp,1), out_channels=[64,128], num_heads=[2,2], num_classes= num_classes(dataset))
model.summary()


from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='accuracy', 
                              patience=10,
                              restore_best_weights=True
                              )
history = model.fit(
            x=Xtrain,
            y=ytrain,
            batch_size=128,
            epochs=250,
            callbacks=[early_stopper],
        )
    
 

############ Create the class Map ####################
# Prepare the data and get the model
del X, Xtrain, Xtest, ytest, ytrain # To free up some space in RAM
gt = y.copy()
X_coh, y = createImageCubes(X2, y, windowSize, removeZeroLabels = False)
X_coh = np.expand_dims(X_coh, axis=4)



model.load_weights('./Models_Weights/'+ dataset +'/HSIFormer.h5')


Y_pred_test = model.predict(X_coh)
y_pred_test = (np.argmax(Y_pred_test, axis=1)).astype(np.uint8)

Y_pred = np.reshape(y_pred_test, gt.shape) + 1

gt_binary = gt

gt_binary[gt_binary>0]=1


new_map = Y_pred*gt_binary

name = 'HSIFormer'
sio.savemat(dataset + '_HSIFormer.mat', {name: new_map})


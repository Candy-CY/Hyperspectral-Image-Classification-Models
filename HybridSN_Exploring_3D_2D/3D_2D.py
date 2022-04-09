import os.path

import keras
import matplotlib.pyplot as plt
from keras.layers import Conv2D,Conv3D,Flatten,Dense,Reshape,BatchNormalization
from keras.layers import Dropout,Input
from keras.models import Model
from keras.optimizers import adam

adam =adam.Adam(lr=0.01,beta_1=0.9,beta_2=0.999,esplon=1e-08)
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,cohen_kappa_score
from operator import truediv
from plotly.offline import init_notebook_mode
import numpy as np
import matplotlib.pyplot as ply
import scipy.io as sio
import io
import spectral
init_notebook_mode(connected=True)

#data loading
dataset = 'IP'
test_ratio = 0.7
windowSize = 25
#读取数据集
def loadData(name):
    data_path = os.path.join(os.getcwd(),'./data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path,'Indian_pines_corrected.mat'))['indan_pnees_corrected']
        labels = sio.loadmat(os.path.join(data_path,'Indan_piines_gt.mat'))['indan_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path,'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path,'Salines_gt.mat'))['salines_gt']
    elif name == ' PU':
        data = sio.loadmat(os.path.join(data_path,'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path,'PaviaU_ft.mat'))['paviaU_gt']
    return data,labels
#划分遥感图像数据集
def splitTrainTestSet(X,y,testRatio,randomState=345):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=testRatio,random_state=randomState,
                                                     stratify=y)
    return X_train,X_test,y_train,y_test
#图像降维
def applyPCA(X,numComPonets=75):
    newX = np.reshape(X,(-1,X.shape[2]))
    pca = PCA(n_components=numComPonets,whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.shape(newX,(X.shape[0],X.shape[1],numComPonets))
    return newX,pca
#数据处理
def padWithZero(X,margin=2):
    newX = np.zeros((X.shape[0]+2*margin,X.shapep[1]+2*margin,X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:x_offset[0]+x_offset,y_offset:x_offset[1]+y_offset,:] = X
    return newX
def createImageCubes(X,y,windowSize=5,removeZeroLabels=True):
    margin = int((windowSize-1)/2)
    zeroPaddedX = padWithZero(X,margin=margin)
    #split patches
    patchesDate = np.zeros((X.shape[0]*X.shape[1],windowSize,windowSize,X.shape[2]))
    patchesLabels = np.zeros((X.shape[0]*X.shape[1]))
    patchIndex = 0
    for r in range(margin,zeroPaddedX.shape[0]-margin):
        for c in range(margin,zeroPaddedX.shape[1]-margin):
            patch = zeroPaddedX[r-margin:r+margin+1,c-margin:c+margin+1]
            patchesDate[patchIndex,:,:,:] = patch
            patchesLabels[patchIndex] = y[r-margin,c-margin]
            patchIndex = patchIndex+1
    if removeZeroLabels:
        patchesDate = patchesDate[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -=1
    return patchesDate,patchesLabels

X,y = loadData(dataset)
K = X.shape[2]
k = 30 if dataset == 'IP' else 15
X,pca = applyPCA(X,numComPonets=K)
X,y = createImageCubes(X,y,windowSize=windowSize)
Xtrain,Xtest,ytrain,ytest = splitTrainTestSet(X,y,test_ratio)
#Model and Training
Xtrain = Xtrain.reshape(-1,windowSize,windowSize,K,1)
ytrain = np_utils.to_categorical(ytrain)
S = windowSize
L = K
output_units = 9 if (dataset=='PU' or  dataset=='PC') else 16
#input layer
input_layer = Input((S,S,L,1))
# convolutional layers
conv_layer1 = Conv3D(filters=8,kernel_size=(3,3,7),activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16,kernel_size=(3,3,5),activation='relu')(conv_layer1)
conv_layer3 = Conv3D(filters=32,kernel_size=(3,3,3),activation='relu')(conv_layer2)
conv3d_shape = conv_layer3.shape
conv_layer3 = Reshape((conv3d_shape[1],conv3d_shape[2],conv3d_shape[3]+conv3d_shape[4]))(conv_layer3)
conv_layer4 = Conv2D(filters=64,kernel_size=(3,3),activation='relu')(conv_layer3)
flatten_layer = Flatten()(conv_layer4)
# fully  connected layers
dense_layer1 = Dense(units=256,activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=128,activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=output_units,activation='softmax')(dense_layer2)
#define the model with input layer and output layer
model = Model(inputs=input_layer,outputs=output_layer)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
#checkpoint
history = model.fit(x=Xtrain,y=ytrain,batch_size=256,epochs=100)
plt.figure(figsize=(7,7))
plt.grid()
plt.plot(history.history['loss'])
plt.figure(figsize=(5,5))
plt.ylim(0,1.1)
plt.grid()
plt.plot(history.history['accuracy'])
#validation
#load best weights
Xtest = Xtest.reshape(-1,windowSize,windowSize,K,1)
ytest = np_utils.to_categorical(ytest)
Y_pred_test = model.predict(Xtest)
y_pred_test = np.argmax(Y_pred_test,axis=1)
classification = classification_report(np.argmax(ytest,axis=1),y_pred_test)
def AA_andEachClassAccuracy(confusion_matrix):
    counter =  confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix,axis=1)
    each_accuracy = np.nan_to_num(truediv(list_diag,list_raw_sum))
    average_accuracy = np.mean(each_accuracy)
    return each_accuracy,average_accuracy
def reports(X_test,y_test,name):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred,axis=1)
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
    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss =  score[0]*100
    Test_accuracy = score[1]*100
    return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100

classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtest,ytest,dataset)
classification = str(classification)
confusion = str(confusion)
file_name = "./classification_report.txt"
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
    x_file.write('{}'.format(confusion))
def Patch(data, height_index, width_index):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch


# load the original image
X, y = loadData(dataset)
height = y.shape[0]
width = y.shape[1]
PATCH_SIZE = windowSize
numComponents = K
X, pca = applyPCA(X, numComponents=numComponents)
X = padWithZero(X, PATCH_SIZE // 2)
# calculate the predicted image
outputs = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        target = int(y[i, j])
        if target == 0:
            continue
        else:
            image_patch = Patch(X, i, j)
            X_test_image = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                               1).astype('float32')
            prediction = (model.predict(X_test_image))
            prediction = np.argmax(prediction, axis=1)
            outputs[i][j] = prediction + 1
ground_truth = spectral.imshow(classes=y, figsize=(7, 7))
predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(7, 7))
spectral.save_rgb("predictions.jpg", outputs.astype(int), colors=spectral.spy_colors)

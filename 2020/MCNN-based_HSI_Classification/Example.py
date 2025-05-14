from operator import truediv
import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization, Lambda
from keras.layers import Dropout, Input
from keras.models import Model
from keras.optimizers import Adam#adam_v2
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import time
from plotly.offline import init_notebook_mode
#from subpixel_conv2d import SubpixelConv2D
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import spectral
#from keras.engine import Layer
from tensorflow.keras.layers import Layer
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects
class SubpixelConv2D(Layer):
    def __init__(self,upsampling_factor=4,**kwargs):
        super(SubpixelConv2D,self).__init__(**kwargs)
        self.upsampling_factor =  upsampling_factor
    def bulid(self,input_shape):
        last_dim = input_shape[-1]
        factor = self.upsampling_factor * self.upsampling_factor
        if last_dim %(factor) != 0:
            raise ValueError('Channel'+str(last_dim)+'should be of'
                            +'integer times of upsampling_factor^2'+str(factor)+'.')
    '''
    def call(self,input,**kwargs):
        return tf.depth_to_space(inputs,self.upsampling_factor)
    '''
    def get_config(self):
        config = {'upsampling_factor':self.upsampling_factor,}
        base_config = super(SubpixelConv2D,self).get_config()
        return dict(list(base_config.items())+ list(config.items()))
    def compute_output_shape(self,input_shape):
        factor = self.upsampling_factor * self.upsampling_factor
        input_shape_1 = None
        if input_shape[1] is not None:
            input_shape_1 = input_shape[1] * self.upsampling_factor
        input_shape_2 = None
        if input_shape[2] is not None:
            input_shape_2 = input_shape[2] * self.upsampling_factor
        dims = [ input_shape[0],
                input_shape_1,
                input_shape_2,
                int(input_shape[3]/factor)
               ]
        return tuple(dims)
get_custom_objects().update({'SubpixelConv2D':SubpixelConv2D})
'''
ip = Input(shape=(32,32,16))
x = SubpixelConv2D(upsampling_factor=4)(ip)
model = Model(ip,x)
model.summary()
model.save('./model.h5')
print('*'*16)
nm = load_model('./model.h5')
print('new model loaded  successfully')
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
## Data
## GLOBAL VARIABLES
dataset = 'IP'
test_ratio = 0.99
train_val_ratio = 1
train_ratio = 1-test_ratio
windowSize = 11
if dataset == 'UP':
    componentsNum = 15
elif dataset == 'UH':
    componentsNum = 50 if test_ratio >= 0.99 else 25
elif dataset == 'IP':
    componentsNum = 110
else:
    componentsNum = 30
drop = 0.4
## define a series of data progress function
def loadData(name):
    data_path = os.path.join(os.getcwd(),'../input/hybridsn/data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'UP':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'UH':
        data = sio.loadmat(os.path.join(data_path, 'HoustonU.mat'))['houstonU'] # 601*2384*50
        labels = sio.loadmat(os.path.join(data_path, 'HoustonU_gt.mat'))['houstonU_gt']
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
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]),dtype="float16")
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
def createPatches(X, y, windowSize=25, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]),dtype="float16")
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]),dtype="float16")
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
X, y = loadData(dataset)
X,pca,ratio = applyPCA(X,numComponents=componentsNum)
X = infoChange(X,componentsNum) # channel-wise shift
X, y = createPatches(X, y, windowSize=windowSize)
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)

## Train
Xtrain = Xtrain.reshape(-1, windowSize, windowSize, componentsNum, 1)
ytrain = np_utils.to_categorical(ytrain)#将类别向量转换为二进制(只有0和1)的矩阵类型表示
Xvalid, Xtest, yvalid, ytest = splitTrainTestSet(Xtest, ytest, (test_ratio-train_ratio/train_val_ratio)/test_ratio)
Xvalid = Xvalid.reshape(-1, windowSize, windowSize, componentsNum, 1)
yvalid = np_utils.to_categorical(yvalid)
if dataset == 'UP':
    output_units = 9
elif dataset == 'UH':
    output_units = 20
else:
    output_units = 16
## implementation of covariance pooling layers
def cov_pooling(features):
    shape_f = features.shape.as_list()
    centers_batch = tf.reduce_mean(tf.transpose(features, [0, 2, 1]),2) # 均值
    centers_batch = tf.reshape(centers_batch, [-1, 1, shape_f[2]])
    centers_batch = tf.tile(centers_batch, [1, shape_f[1], 1]) # 张量扩展
    tmp = tf.subtract(features, centers_batch)
    tmp_t = tf.transpose(tmp, [0, 2, 1])
    features_t = 1/tf.cast((shape_f[1]-1),tf.float32)*tf.matmul(tmp_t, tmp)
    trace_t = tf.linalg.trace(features_t)
    trace_t = tf.reshape(trace_t, [-1, 1])
    trace_t = tf.tile(trace_t, [1, shape_f[2]])
    trace_t = 0.0001 * tf.matrix_diag(trace_t)
    return tf.add(features_t,trace_t)
def feature_vector(features):
    # features，是对称的，由于张量无法像矩阵一样直接取上三角数据拉成一维向量
    shape_f = features.shape.as_list()
    feature_upper = tf.linalg.band_part(features,0,shape_f[2])
    return feature_upper
## input layer
input_layer = Input((windowSize, windowSize, componentsNum, 1))
## convolutional layers
conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv_layer1)
conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(conv_layer2)
# print(conv_layer3._keras_shape)
#conv3d_shape = conv_layer3._keras_shape
conv3d_shape = conv_layer3.shape
conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(conv_layer3)
# conv2d_shape = conv_layer4._keras_shape
# conv_layer4 = SubpixelConv2D(upsampling_factor=8)(conv_layer4)
#conv2d_shape = conv_layer4._keras_shape
conv2d_shape = conv_layer4.shape
conv_layer4 = Reshape((conv2d_shape[1] * conv2d_shape[2], conv2d_shape[3]))(conv_layer4)
#conv2d_shape = conv_layer4._keras_shape
conv2d_shape = conv_layer4.shape
cov_pooling_layer1 = Lambda(cov_pooling,output_shape=(conv2d_shape[2],conv2d_shape[2]),mask=None,arguments=None)(conv_layer4)
cov_pooling_layer2 = Lambda(feature_vector,output_shape=(conv2d_shape[2],conv2d_shape[2]),mask=None,arguments=None)(cov_pooling_layer1)
# (int)(cov_pooling_shape[1]*(cov_pooling_shape[2]+1)/2)为有效参数数量
flatten_layer = Flatten()(cov_pooling_layer2)
## fully connected layers
dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)
# define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()
# compiling the model adam_v2.
adam = Adam(lr=0.001, decay=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# checkpoint
filepath = "./best-model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
start = time.time()
history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=100, validation_data=(Xvalid,yvalid), callbacks=callbacks_list)  #,validation_split=(1/3)
end = time.time()
print((end - start)/60)
plt.figure(figsize=(7,7))
plt.grid()
plt.plot(history.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training','Validation'], loc='upper right')
plt.savefig("loss_curve.pdf")
plt.show()
plt.figure(figsize=(5,5))
plt.ylim(0,1.1)
plt.grid()
plt.plot(history.history['acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training','Validation'])
plt.savefig("acc_curve.pdf")
plt.show()

## Test
# load best weights
model.load_weights("./best-model.hdf5")
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
Xtest = Xtest.reshape(-1, windowSize, windowSize, componentsNum, 1)
# Xtest.shape
ytest = np_utils.to_categorical(ytest)
# ytest.shape
Y_pred_test = model.predict(Xtest)
y_pred_test = np.argmax(Y_pred_test, axis=1)
classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
print(classification)
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
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'UP':
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']
    elif name == 'UH':
        target_names = ['Healthy grass','Stressed grass','Artificial turf','Evergreen trees', 'Deciduous trees','Bare earth','Water',
                        'Residential buildings','Non-residential buildings','Roads','Sidewalks','Crosswalks','Major thoroughfares','Highways',
                       'Railways','Paved parking lots','Unpaved parking lots','Cars','Trains','Stadium seats']

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
confusion1 = str(confusion)
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
    x_file.write('{}'.format(confusion1))
## Run Data
def Patch(data,height_index,width_index):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch
X, y = loadData(dataset)
height = y.shape[0]
width = y.shape[1]
PATCH_SIZE = windowSize
X,pca,ratio = applyPCA(X,numComponents=componentsNum)
X = infoChange(X,componentsNum)
X = padWithZeros(X, PATCH_SIZE//2)
# calculate the predicted image
outputs = np.zeros((height,width),dtype="float16")
outputs2 = np.zeros((height,width),dtype="float16")
for i in range(height):
    for j in range(width):
        target = int(y[i,j])
        if target == 0 :
            image_patch=Patch(X,i,j)
            X_test_image = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2], 1).astype('float32')
            prediction2 = (model.predict(X_test_image))
            prediction2 = np.argmax(prediction2, axis=1)
            outputs2[i][j] = prediction2+1
        else :
            image_patch=Patch(X,i,j)
            X_test_image = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2], 1).astype('float32')
            prediction = (model.predict(X_test_image))
            prediction = np.argmax(prediction, axis=1)
            outputs[i][j] = prediction+1
            outputs2[i][j] = prediction+1
import spectral
ground_truth = spectral.imshow(classes = y,figsize =(7,7))
predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(7,7))
predict_image2 = spectral.imshow(classes = outputs2.astype(int),figsize =(7,7))
spectral.save_rgb("./predictions.png", outputs.astype(int), colors=spectral.spy_colors)
spectral.save_rgb("./predictions2.png", outputs2.astype(int), colors=spectral.spy_colors)


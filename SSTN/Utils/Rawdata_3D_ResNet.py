# -*- coding: utf-8 -*-
"""
Train a simple deep ResNet on the UPavia 2D dataset.

Exception: Only layers of same output shape can be merged using sum mode. Layer shapes:
[(None, 7, 7, 128), (None, 4, 4, 128)]
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
import keras.callbacks as kcallbacks
from sklearn import metrics
import time
import resnet
import zeroPadding
import normalization
import sampleFixNum
import doPCA
import collections
import time
import averageAccuracy


batch_size = 32                                                                #training parameters setting
nb_classes = 9
nb_epoch = 400                #400

# input image dimensions
img_rows, img_cols = 27, 27           #27, 27
# The CIFAR10 images are RGB.
INPUT_DIMENSION = 103

VALIDATION_SPLIT = 0.9
PATCH_LENGTH = 13                   #Patch_size (13*2+1)*(13*2+1)
#INPUT_DIMENSION = 103

best_weights_path = '/home/finoa/DL-on-HSI-Classification/Best_models/best_3DRAWResNet_UPavia.hdf5'
uPavia = sio.loadmat('/home/finoa/DL-on-HSI-Classification/Datasets/UPavia/PaviaU.mat')
gt_uPavia = sio.loadmat('/home/finoa/DL-on-HSI-Classification/Datasets/UPavia/PaviaU_gt.mat')
data_UP = uPavia['paviaU']
gt_UP = gt_uPavia['paviaU_gt']
print(data_UP.shape)

# def zeroPadding(old_matrix, pad_length):
#     new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length),(pad_length, pad_length)), 'constant', constant_values=0)
#     return new_matrix

def trainDataZeroPadding(old_matrx, pad_length):
    new_matrix = np.lib.pad(old_matrx, ((0, pad_length), (0, pad_length)), 'constant', constant_values=0)
    return new_matrix

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :, :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1), :]
    return selected_patch

# def sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
#     labels_loc = {}
#     train = {}
#     test = {}
#     m = max(groundTruth)
#     for i in range(m):
#         indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
#         np.random.shuffle(indices)
#         labels_loc[i] = indices
#         nb_val = int(proptionVal * len(indices))
#         train[i] = indices[:-nb_val]
#         test[i] = indices[-nb_val:]
# #    whole_indices = []
#     train_indices = []
#     test_indices = []
#     for i in range(m):
# #        whole_indices += labels_loc[i]
#         train_indices += train[i]
#         test_indices += test[i]
#     np.random.shuffle(train_indices)
#     np.random.shuffle(test_indices)
#     return train_indices, test_indices

data = data_UP.reshape(np.prod(data_UP.shape[:2]),np.prod(data_UP.shape[2:]))
gt = gt_UP.reshape(np.prod(gt_UP.shape[:2]),)

data = normalization.Normalization(data)

data_ = data.reshape(data_UP.shape[0], data_UP.shape[1],data_UP.shape[2])
# data_trans = data.transpose()
# whole_pca = doPCA.dimension_PCA(data_trans, data_UP, INPUT_DIMENSION)
whole_pca = data_ #raw data

print (whole_pca.shape)

padded_data = zeroPadding.zeroPadding_3D(whole_pca, PATCH_LENGTH)

ITER = 10                               #ITER = 10
CATEGORY = 9
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CATEGORY))

for index_iter in range(ITER):
    print ("Iteration #:", index_iter)
    millis = int(round(time.time()) * 1000) % 4294967295
    np.random.seed(millis)

    train_indices, test_indices = sampleFixNum.sampling(VALIDATION_SPLIT, gt)

    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    #first principal component training data
    train_assign = indexToAssignment(train_indices, whole_pca.shape[0], whole_pca.shape[1], PATCH_LENGTH)
    train_data = np.zeros((len(train_assign), 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION))
    #train_data = np.zeros((len(train_assign), 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data,train_assign[i][0],train_assign[i][1],PATCH_LENGTH)
    #    train_data[i] = trainDataZeroPadding(train_data_origin, img_rows - 2*PATCH_LENGTH - 1)

    #first principal component testing data
    test_assign = indexToAssignment(test_indices, whole_pca.shape[0], whole_pca.shape[1], PATCH_LENGTH)
    test_data = np.zeros((len(test_assign), 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION))
    #test_data = np.zeros((len(test_assign), 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1))
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data,test_assign[i][0],test_assign[i][1],PATCH_LENGTH)
    #    test_data[i] = trainDataZeroPadding(test_data_origin, img_cols - 2*PATCH_LENGTH - 1)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    model = resnet.ResnetBuilder.build_resnet_50((INPUT_DIMENSION, img_rows, img_cols), nb_classes)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    earlyStopping = kcallbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
    saveBestModel = kcallbacks.ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    tic = time.clock()
    history = model.fit(x_train, y_train, validation_data=(x_test[-4800:], y_test[-4800:]),  batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True ,callbacks=[earlyStopping, saveBestModel])
    toc = time.clock()

    tic1 = time.clock()
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
    toc1 = time.clock()

    print('Training Time: ', toc - tic)
    print('Test time:', toc1 - tic1)
    print('Test score:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])
    print(history.history.keys())

    pred_test = model.predict(x_test).argmax(axis=1)
    collections.Counter(pred_test)
    gt_test = gt[test_indices] - 1
    overall_acc = metrics.accuracy_score(pred_test, gt_test)
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test)
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    OA.append(overall_acc)
    AA.append(average_acc)
    TRAINING_TIME.append(toc - tic)
    TESTING_TIME.append(toc1 - tic1)
    ELEMENT_ACC[index_iter, :] = each_acc

    print("Overall Accuracy:", overall_acc)

    print("Confusion matrix:", confusion_matrix)

    print("Average Accuracy:", average_acc)

    print("Each Class Accuracies are listed as follows:")
    for idx, acc in enumerate(each_acc):
        print("Class %d : %.3e" % (idx, acc))


f = open('/home/finoa/Desktop/record_new_3DRes_32.txt', 'w')

sentence1 = 'OAs, mean_OA ± std_OA for each iteration are:' + str(OA) + str(np.mean(OA)) + ' ± ' + str(np.std(OA)) +'\n'
f.write(sentence1)
sentence2 = 'AAs, mean_AA ± std_AA for each iteration are:' + str(AA) + str(np.mean(AA)) + ' ± ' + str(np.std(AA)) +'\n'
f.write(sentence2)
sentence3 = 'Average Training time is :' + str(np.mean(TRAINING_TIME)) +'\n'
f.write(sentence3)
sentence4 = 'Average Testing time is:' + str(np.mean(TESTING_TIME)) +'\n'
f.write(sentence4)

element_mean = np.mean(ELEMENT_ACC, axis=0)
element_std = np.std(ELEMENT_ACC, axis=0)
sentence5 = "Mean of all elements in confusion matrix:" + str(np.mean(ELEMENT_ACC, axis=0)) +'\n'
f.write(sentence5)
sentence6 = "Standard deviation of all elements in confusion matrix" + str(np.std(ELEMENT_ACC, axis=0)) +'\n'
f.write(sentence6)
f.close()


print_matrix = np.zeros((CATEGORY), dtype=object)
for i in range(CATEGORY):
    print_matrix[i] = str(element_mean[i]) + " ± " + str(element_std[i])

np.savetxt("/home/finoa/Desktop/element_acc_raw3d_resnet.txt", print_matrix.astype(str), fmt='%s', delimiter="\t", newline='\n')


print('Test score:', loss_and_metrics[0])
print('Test accuracy:', loss_and_metrics[1])
print(history.history.keys())

plt.figure(2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
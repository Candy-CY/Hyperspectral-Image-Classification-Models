# coding: utf-8

import os
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
set_session(session)

import matplotlib

matplotlib.use('Agg')

from tensorflow.keras.optimizers import RMSprop
import numpy as np

from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, MaxPool2D, BatchNormalization, Conv1D
from tensorflow.keras.layers import Input, Dense, Activation, Add, LSTM, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, \
    Lambda
from tensorflow.keras.models import Model, Sequential, save_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Concatenate, RepeatVector
from tensorflow.keras import backend as K

from myTCN import TCN
from HyperFunctions import *
import time
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def Spe(time_step, nb_features):
    LSTMInput = Input(shape=(time_step, nb_features), name='LSTMInput')

    LSTMSpectral = LSTM(176, name='LSTMSpectral', consume_less='gpu', W_regularizer=l2(0.0001),
                        U_regularizer=l2(0.0001))(LSTMInput)

    LSTMDense = Dense(176, activation='relu', name='LSTMDense')(LSTMSpectral)

    LSTMSoftmax = Dense(nb_classes, activation='softmax')(LSTMDense)

    model = Model(inputs=LSTMInput, outputs=LSTMSoftmax)

    intermediate_tensor_function = K.function([model.layers[0].input], [model.layers[2].output])
    intermediate_tensor = intermediate_tensor_function([X_test])[0]
    # print("intermediate_tensor: ", intermediate_tensor.shape)  # (4872, 112)

    model.summary()

    rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=1e-05)

    model.compile(optimizer=rmsp, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def Spa(w_row, w_col, num_PC):
    HybridInput = Input(shape=[w_row, w_col, num_PC])

    CONV111 = Conv2D(16, (9, 9), activation='relu', padding='same')(HybridInput)
    CONV111 = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None,
                                 )(CONV111)
    POOL111 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CONV111)
    CONV112 = Conv2D(16, (9, 9), activation='relu', padding='same')(POOL111)
    CONV112 = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None,
                                 )(CONV112)
    POOL112 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CONV112)
    CONV113 = Conv2D(16, (9, 9), activation='relu', padding='same')(POOL112)
    CONV113 = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None,
                                 )(CONV113)
    POOL113 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CONV113)

    HybridFlatten1 = Flatten()(POOL113)
    HybridDense1 = Dense(units=128, activation='relu')(HybridFlatten1)  # 128

    CONV21 = Conv2D(32, (7, 7), activation='relu', padding='same')(HybridInput)
    CONV21 = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None,
                                )(CONV21)
    CONV21 = Concatenate(axis=-1)([CONV111, CONV21])
    POOL21 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CONV21)
    CONV22 = Conv2D(32, (7, 7), activation='relu', padding='same')(POOL21)
    CONV22 = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None,
                                )(CONV22)
    CONV22 = Concatenate(axis=-1)([CONV112, CONV22])
    POOL22 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CONV22)
    CONV23 = Conv2D(32, (7, 7), activation='relu', padding='same')(POOL22)
    CONV23 = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None,
                                )(CONV23)
    CONV23 = Concatenate(axis=-1)([CONV113, CONV23])
    POOL23 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CONV23)

    HybridFlatten2 = Flatten()(POOL23)
    HybridDense2 = Dense(units=128, activation='relu')(HybridFlatten2)  # 128

    CONV31 = Conv2D(64, (5, 5), activation='relu', padding='same')(HybridInput)
    CONV31 = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None,
                                )(CONV31)
    CONV31 = Concatenate(axis=-1)([CONV21, CONV31])
    POOL31 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CONV31)
    CONV32 = Conv2D(64, (5, 5), activation='relu', padding='same')(POOL31)
    CONV32 = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None,
                                )(CONV32)
    CONV32 = Concatenate(axis=-1)([CONV22, CONV32])
    POOL32 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CONV32)
    CONV33 = Conv2D(64, (5, 5), activation='relu', padding='same')(POOL32)
    CONV33 = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None,
                                )(CONV33)
    CONV33 = Concatenate(axis=-1)([CONV23, CONV33])
    POOL33 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CONV33)

    HybridFlatten3 = Flatten()(POOL33)
    HybridDense3 = Dense(units=128, activation='relu')(HybridFlatten3)  # 128

    CONV41 = Conv2D(128, (3, 3), activation='relu', padding='same')(HybridInput)
    CONV41 = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None,
                                )(CONV41)
    CONV41 = Concatenate(axis=-1)([CONV31, CONV41])
    POOL41 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CONV41)
    CONV42 = Conv2D(128, (3, 3), activation='relu', padding='same')(POOL41)
    CONV42 = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None,
                                )(CONV42)
    CONV42 = Concatenate(axis=-1)([CONV32, CONV42])
    POOL42 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CONV42)
    CONV43 = Conv2D(128, (3, 3), activation='relu', padding='same')(POOL42)
    CONV43 = BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9, weights=None,
                                )(CONV43)
    CONV43 = Concatenate(axis=-1)([CONV33, CONV43])
    POOL43 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CONV43)

    HybridFlatten4 = Flatten()(POOL43)
    HybridDense4 = Dense(units=128, activation='relu')(HybridFlatten4)  # 128

    JOINTCNN = Concatenate(axis=-1)([HybridDense1, HybridDense2, HybridDense3, HybridDense4])

    JOINTsigmoid = Dense(units=512, activation='sigmoid')(JOINTCNN)  # 128
    JOINTCNN = Lambda(lambda inputs: JOINTsigmoid * inputs)(JOINTCNN)

    JOINTDENSECNN = Dense(128, activation='relu')(JOINTCNN)
    JOINTSOFTMAXCNN = Dense(units=nb_classes, activation='softmax')(JOINTDENSECNN)

    model = Model(inputs=[HybridInput], outputs=[JOINTSOFTMAXCNN])

    rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=1e-05)

    model.compile(optimizer=rmsp, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def OSICN(time_step, nb_features, w_row, w_col, num_PC):
    LSTMInput = Input(shape=(time_step, nb_features), name='LSTMInput')

    LSTMSpectral = LSTM(128, kernel_regularizer=l2(0.0001),
                        bias_regularizer=l2(0.0001))(LSTMInput)
    LSTMSpectral2 = LSTM(160, kernel_regularizer=l2(0.0001),
                         bias_regularizer=l2(0.0001))(LSTMInput)
    LSTMSpectral3 = LSTM(192, kernel_regularizer=l2(0.0001),
                         bias_regularizer=l2(0.0001))(LSTMInput)
    LSTMSpectral4 = LSTM(224, kernel_regularizer=l2(0.0001),
                         bias_regularizer=l2(0.0001))(LSTMInput)

    LSTMDense = Dense(128, activation='relu')(LSTMSpectral)
    LSTMDense2 = Dense(128, activation='relu')(LSTMSpectral2)
    LSTMDense3 = Dense(128, activation='relu')(LSTMSpectral3)
    LSTMDense4 = Dense(128, activation='relu')(LSTMSpectral4)
    LSTMDense = Concatenate(axis=-1)([LSTMDense, LSTMDense2, LSTMDense3, LSTMDense4])

    LSTMSOFTMAX = Dense(nb_classes, activation='softmax', name='LSTMSOFTMAX')(LSTMDense)

    CNNInput = Input(shape=[w_row, w_col, num_PC], name='CNNInput')

    CONV11 = Conv2D(32, (9, 9), activation='relu', padding='same', name='CONV11')(CNNInput)
    POOL11 = MaxPooling2D((2, 2), name='POOL11')(CONV11)
    CONV12 = Conv2D(128, (9, 9), activation='relu', padding='same', name='CONV12')(POOL11)
    POOL12 = MaxPooling2D((2, 2), name='POOL12')(CONV12)
    # POOL12 = Add()([LSTMSpectral, POOL12])
    CONV13 = Conv2D(128, (9, 9), activation='relu', padding='same', name='CONV13')(POOL12)
    POOL13 = MaxPooling2D((2, 2), name='POOL13')(CONV13)
    POOL13 = Add()([LSTMSpectral, POOL13])

    CONV21 = Conv2D(32, (7, 7), activation='relu', padding='same')(CNNInput)
    POOL21 = MaxPooling2D((2, 2))(CONV21)
    CONV22 = Conv2D(64, (7, 7), activation='relu', padding='same')(POOL21)
    CONV22 = Concatenate(axis=-1)([CONV12, CONV22])
    POOL22 = MaxPooling2D((2, 2))(CONV22)
    # POOL22 = Add()([LSTMSpectral2, POOL22])
    CONV23 = Conv2D(32, (7, 7), activation='relu', padding='same')(POOL22)
    CONV23 = Concatenate(axis=-1)([CONV13, CONV23])
    POOL23 = MaxPooling2D((2, 2))(CONV23)
    POOL23 = Add()([LSTMSpectral2, POOL23])

    CONV31 = Conv2D(32, (5, 5), activation='relu', padding='same')(CNNInput)
    POOL31 = MaxPooling2D((2, 2))(CONV31)
    CONV32 = Conv2D(64, (5, 5), activation='relu', padding='same')(POOL31)
    CONV32 = Concatenate(axis=-1)([CONV22, CONV32])
    POOL32 = MaxPooling2D((2, 2))(CONV32)
    # POOL22 = Add()([LSTMSpectral2, POOL22])
    CONV33 = Conv2D(32, (5, 5), activation='relu', padding='same')(POOL32)
    CONV33 = Concatenate(axis=-1)([CONV23, CONV33])
    POOL33 = MaxPooling2D((2, 2))(CONV33)
    POOL33 = Add()([LSTMSpectral3, POOL33])

    CONV41 = Conv2D(32, (3, 3), activation='relu', padding='same')(CNNInput)
    POOL41 = MaxPooling2D((2, 2))(CONV41)
    CONV42 = Conv2D(64, (3, 3), activation='relu', padding='same')(POOL41)
    CONV42 = Concatenate(axis=-1)([CONV32, CONV42])
    POOL42 = MaxPooling2D((2, 2))(CONV42)
    # POOL22 = Add()([LSTMSpectral2, POOL22])
    CONV43 = Conv2D(32, (3, 3), activation='relu', padding='same')(POOL42)
    CONV43 = Concatenate(axis=-1)([CONV33, CONV43])
    POOL43 = MaxPooling2D((2, 2))(CONV43)
    POOL43 = Add()([LSTMSpectral4, POOL43])

    FLATTEN11 = Flatten(name='FLATTEN11')(POOL11)
    FLATTEN12 = Flatten(name='FLATTEN12')(POOL12)
    FLATTEN13 = Flatten(name='FLATTEN13')(POOL13)
    FLATTEN21 = Flatten(name='FLATTEN21')(POOL21)
    FLATTEN22 = Flatten(name='FLATTEN22')(POOL22)
    FLATTEN23 = Flatten(name='FLATTEN23')(POOL23)
    FLATTEN31 = Flatten(name='FLATTEN31')(POOL31)
    FLATTEN32 = Flatten(name='FLATTEN32')(POOL32)
    FLATTEN33 = Flatten(name='FLATTEN33')(POOL33)
    FLATTEN41 = Flatten(name='FLATTEN41')(POOL41)
    FLATTEN42 = Flatten(name='FLATTEN42')(POOL42)
    FLATTEN43 = Flatten(name='FLATTEN43')(POOL43)

    DENSE11 = Dense(128, activation='relu')(FLATTEN11)
    DENSE12 = Dense(128, activation='relu')(FLATTEN12)
    DENSE13 = Dense(128, activation='relu')(FLATTEN13)
    DENSE21 = Dense(128, activation='relu')(FLATTEN21)
    DENSE22 = Dense(128, activation='relu')(FLATTEN22)
    DENSE23 = Dense(128, activation='relu')(FLATTEN23)
    DENSE31 = Dense(128, activation='relu')(FLATTEN31)
    DENSE32 = Dense(128, activation='relu')(FLATTEN32)
    DENSE33 = Dense(128, activation='relu')(FLATTEN33)
    DENSE41 = Dense(128, activation='relu')(FLATTEN41)
    DENSE42 = Dense(128, activation='relu')(FLATTEN42)
    DENSE43 = Dense(128, activation='relu')(FLATTEN43)

    CNNDense = Concatenate(axis=-1)(
        [DENSE11, DENSE12, DENSE13, DENSE21, DENSE22, DENSE23, DENSE31, DENSE32, DENSE33, DENSE41, DENSE42, DENSE43])

    CNNSOFTMAX = Dense(nb_classes, activation='softmax', name='CNNSOFTMAX')(CNNDense)

    JOINT = Concatenate()([LSTMDense, CNNDense])
    JOINTDENSE = Dense(128, activation='relu', name='JOINTDENSE')(JOINT)

    JOINTSOFTMAX = Dense(nb_classes, activation='softmax', name='JOINTSOFTMAX')(JOINTDENSE)

    model = Model(inputs=[LSTMInput, CNNInput], outputs=[JOINTSOFTMAX, LSTMSOFTMAX, CNNSOFTMAX])

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'], loss_weights=[1, 1, 1])

    return model



storage_location = './OSICN/'
if not os.path.isdir(storage_location):
    os.makedirs(storage_location)

w = 28
num_PC = 4
israndom = True
s1s2 = 2
time_step = 3

dataID = 1
nb_classes = 9
randtime = 10
batch_size = 128
nb_epoch = 200

# # ************************************* Spectral **************************************#
# OASpectral = np.zeros((nb_classes + 2, randtime))
# TimeSpe = np.zeros((2, randtime))
# for r in range(0, randtime):
#     data = HyperspectralSamples(dataID=dataID, timestep=time_step, w=w, num_PC=num_PC, israndom=israndom, s1s2=s1s2)
#     X = data[0]
#     X_train = data[1]
#     X_test = data[2]
#     XP = data[3]
#     XP_train = data[4]
#     XP_test = data[5]
#     Y = data[6] - 1
#     Y_train = data[7] - 1
#     Y_test = data[8] - 1
#
#
#     nb_classes = Y_train.max() + 1
#
#     nb_feature_perTime = X.shape[-1]
#
#     img_rows, img_cols = XP.shape[-1], XP.shape[-1]
#     y_train = to_categorical(Y_train, nb_classes)
#     y_test = to_categorical(Y_test, nb_classes)
#
#     model = Spa(w_row = w, w_col = w, num_PC = num_PC)
#
#     # model.summary()
#
#     filepath = os.path.join(os.getcwd(), storage_location + repr(dataID) + '_r' + repr(r+1) + '_TCNbest_model.hdf5')
#     checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='max')  # val_acc
#
#     callbacks_list = [checkpoint]
#
#     tic0 = time.time()
#     history = model.fit([XP_train], [y_train], epochs=nb_epoch, batch_size=batch_size, shuffle=True, verbose=0)
#     toc0 = time.time()
#
#     tic1 = time.time()
#     Map = model.predict([XP], verbose=0)
#     toc1 = time.time()
#
#     losses = history.history
#
#     PredictLabel = model.predict([XP_test], verbose=0).argmax(axis=-1)
#
#     OA, Kappa, ProducerA = CalAccuracy(PredictLabel, Y_test[:, 0])
#     OASpectral[0:nb_classes, r] = ProducerA * 100
#     OASpectral[-2, r] = OA * 100
#     OASpectral[-1, r] = Kappa * 100
#
#     TimeSpe[0, r] = toc0 - tic0
#     TimeSpe[1, r] = toc1 - tic1
#     print('rand', r + 1, '_dataID_', dataID, '_CNN OA: %f' % (OA * 100) + '\n')
#
#
#     file_name = (storage_location + str(dataID) + 'r' +repr(r+1) + '_' + repr(int(OA*10000)) + '_CNN_Report.txt')
#     with open(file_name, 'w') as x_file:
#         x_file.write('{} OA (%)'.format(OA * 100))
#         x_file.write('\n')
#         x_file.write('\n')
#         x_file.write('{} Kappa (%)'.format(Kappa * 100))
#         x_file.write('\n')
#         x_file.write('\n')
#         x_file.write('{} Producer Accuracy (%)'.format(ProducerA * 100))
#         x_file.write('\n')
#         x_file.write('\n')
#         x_file.write('{} Train time '.format(toc0 - tic0))
#         x_file.write('\n')
#         x_file.write('\n')
#         x_file.write('{} Test time '.format(toc1 - tic1))
#
#
#     Spectral = Map.argmax(axis=-1)
#
#     X_result = DrawResult(Spectral, dataID)
#
#     plt.imsave(storage_location + repr(dataID) +'_CNN_r' + repr(r + 1) + 'OA_' + repr(
#         int(OA * 10000)) + '.png', X_result)
#
#
#     if (r == (randtime - 1)):
#         OAmean = np.mean(OASpectral[-2])
#         OA_all_std = np.std(OASpectral, axis=1, ddof=0)
#         OAstd = OA_all_std[-2]
#         Kappamean = np.mean(OASpectral[-1])
#         Kappastd = OA_all_std[-1]
#         ProducerAmean = np.mean(OASpectral[0:nb_classes], axis=1)
#         ProducerAstd = OA_all_std[0:nb_classes]
#         Timemean_tra = np.mean(TimeSpe[0])
#         Timestd = np.std(TimeSpe, axis=1, ddof=0)
#         Timestd_tra = Timestd[0]
#         Timemean_tes = np.mean(TimeSpe[1])
#         Timestd_tes = Timestd[1]
#         file_name = (storage_location + str(dataID) + '_Mean_CNN_ClassificationReport.txt')
#         with open(file_name, 'w') as x_file:
#             x_file.write('{} OA mean (%)'.format(OAmean))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} OA std'.format(OAstd))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Kappa mean (%)'.format(Kappamean))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Kappa std'.format(Kappastd))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Producer Accuracy mean (%)'.format(ProducerAmean))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Producer Accuracy std'.format(ProducerAstd))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Time mean(train) '.format(Timemean_tra))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Time mean(train) std '.format(Timestd_tra))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Time mean(test) '.format(Timemean_tes))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Time mean(test) std '.format(Timestd_tes))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Array for all '.format(OASpectral))
#
# print('**********************   CNN Time(train) is: %6.2f' % (toc0 - tic0) + 's.   **********************\n\n')



# # # # ****************************************** Spatial *******************************************#
# w = 28
# num_PC = 4
# israndom = True
#
# OASpatial = np.zeros((nb_classes + 2, randtime))
# TimeSpa = np.zeros((2, randtime))
# for r in range(0, randtime):
#     data = HyperspectralSamples(dataID=dataID, w=w, num_PC=num_PC, israndom=israndom)
#     X = data[0]
#     X_train = data[1]
#     X_test = data[2]
#     XP = data[3]
#     XP_train = data[4]
#     XP_test = data[5]
#     Y = data[6] - 1
#     Y_train = data[7] - 1
#     Y_test = data[8] - 1
#
#     nb_classes = Y_train.max() + 1
#
#     nb_feature_perTime = X.shape[-1]
#
#     img_rows, img_cols = XP.shape[-1], XP.shape[-1]
#
#     y_train = to_categorical(Y_train, nb_classes)
#     y_test = to_categorical(Y_test, nb_classes)
#
#     filepath = os.path.join(os.getcwd(),
#                             storage_location + repr(dataID) + '_r' + repr(r + 1) + '_w' + repr(w) + 'PC' + repr(
#                                 num_PC) + '_MLCNNbest_model.hdf5')
#     checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True,
#                                  mode='max')
#
#     tbCallBack = TensorBoard(log_dir="./logs/")
#
#     callbacks_list = [checkpoint, tbCallBack]
#
#     XP_train = XP_train.reshape(-1, w, w, num_PC)
#
#     model = Spa(w_row=w, w_col=w, num_PC=num_PC)
#
#     tic0 = time.time()
#     history = model.fit(x=XP_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=0)
#
#     toc0 = time.time()
#
#     XP_test = XP_test.reshape(-1, w, w, num_PC)
#
#     tic1 = time.time()
#     Ypre_test = model.predict(XP_test, verbose=0)
#     toc1 = time.time()
#
#     ypre_test = np.argmax(Ypre_test, axis=1)
#
#     score = model.evaluate(XP_test, y_test)
#     TestLoss = score[0]
#     TestAcc = score[1]
#
#     OA, Kappa, ProducerA = CalAccuracy(ypre_test, Y_test[:, 0])
#
#     OASpatial[0:nb_classes, r] = ProducerA * 100
#     OASpatial[-2, r] = OA * 100
#     OASpatial[-1, r] = Kappa * 100
#
#     TimeSpa[0, r] = toc0 - tic0
#     TimeSpa[1, r] = toc1 - tic1
#     print(('rand', r + 1, '_dataID', dataID, '_MLCNN OA ', OA * 100))
#
#     file_name = (storage_location + str(dataID) + 'r' + repr(r + 1) + '_w' + repr(w) + 'PC' + repr(num_PC) + '_' + repr(
#         int(OA * 10000)) + '_MLCNN_Report.txt')
#     with open(file_name, 'w') as x_file:
#         x_file.write('{} OA (%)'.format(OA * 100))
#         x_file.write('\n')
#         x_file.write('\n')
#         x_file.write('{} Kappa (%)'.format(Kappa * 100))
#         x_file.write('\n')
#         x_file.write('\n')
#         x_file.write('{} Producer Accuracy (%)'.format(ProducerA * 100))
#         x_file.write('\n')
#         x_file.write('\n')
#         x_file.write('{} Train time '.format(toc0 - tic0))
#         x_file.write('\n')
#         x_file.write('\n')
#         x_file.write('{} Test time '.format(toc1 - tic1))
#
#     XP = XP.reshape(-1, w, w, num_PC)
#     Map = model.predict([XP], verbose=0)
#
#     Spatial = Map.argmax(axis=-1)
#
#     X_result = DrawResult(Spatial, dataID)
#
#     plt.imsave(storage_location + repr(dataID) + '_MLCNN_r' + repr(r + 1) + '_w' + repr(w) + 'PC' + repr(
#         num_PC) + 'OA_' + repr(int(OA * 10000)) + '.png',
#                X_result)
#
#     if (r == (randtime - 1)):
#         OAmean = np.mean(OASpatial[-2])
#         OA_all_std = np.std(OASpatial, axis=1, ddof=0)  # ddof=0: 总体偏差
#         OAstd = OA_all_std[-2]
#         Kappamean = np.mean(OASpatial[-1])
#         Kappastd = OA_all_std[-1]
#         ProducerAmean = np.mean(OASpatial[0:nb_classes], axis=1)
#         ProducerAstd = OA_all_std[0:nb_classes]
#         Timemean_tra = np.mean(TimeSpa[0])
#         Timestd = np.std(TimeSpa, axis=1, ddof=0)
#         Timestd_tra = Timestd[0]
#         Timemean_tes = np.mean(TimeSpa[1])
#         Timestd_tes = Timestd[1]
#         file_name = (storage_location + str(dataID) + '_Mean_MLCNN' + '_w' + repr(w) + 'PC' + repr(num_PC) + repr(
#             OA * 100) + '_ClassificationReport.txt')
#         with open(file_name, 'w') as x_file:
#             x_file.write('{} OA mean (%)'.format(OAmean))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} OA std'.format(OAstd))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Kappa mean (%)'.format(Kappamean))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Kappa std'.format(Kappastd))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Producer Accuracy mean (%)'.format(ProducerAmean))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Producer Accuracy std'.format(ProducerAstd))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Time mean(train) '.format(Timemean_tra))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Time mean(train) std '.format(Timestd_tra))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Time mean(test) '.format(Timemean_tes))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Time mean(test) std '.format(Timestd_tes))
#             x_file.write('\n')
#             x_file.write('\n')
#             x_file.write('{} Array for all '.format(OASpatial))
#
# print('**********************   Time(train) is: %6.2f' % (toc0 - tic0) + 's.   **********************\n\n')



# ************************************* joint OSICN **************************************#
OAJoint = np.zeros((nb_classes + 2, randtime))
TimeJoint = np.zeros((2, randtime))
for w in range(44, 64, 4):
    for num_PC in range(8, 17, 3):
        for r in range(0, randtime):
            data = HyperspectralSamples(dataID=dataID, timestep=time_step, w=w, num_PC=num_PC, israndom=israndom,
                                        s1s2=s1s2)
            X = data[0]
            X_train = data[1]
            X_test = data[2]
            XP = data[3]
            XP_train = data[4]
            XP_test = data[5]
            Y = data[6] - 1
            Y_train = data[7] - 1
            Y_test = data[8] - 1

            nb_classes = Y_train.max() + 1

            nb_feature_perTime = X.shape[-1]

            img_rows, img_cols = XP.shape[-1], XP.shape[-1]

            y_train = to_categorical(Y_train, nb_classes)
            y_test = to_categorical(Y_test, nb_classes)

            model = OSICN(time_step=time_step, nb_features=nb_feature_perTime, w_row=w, w_col=w, num_PC=num_PC)

            XP_train = XP_train.reshape(-1, w, w, num_PC)

            filepath = os.path.join(os.getcwd(),
                                    storage_location + repr(dataID) + '_r' + repr(r + 1) + '_w' + repr(w) + 'PC' + repr(
                                        num_PC) + '_Jointbest_model.hdf5')
            checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True,
                                         save_weights_only=True, mode='max')

            callbacks_list = [checkpoint]

            tic0 = time.time()
            history = model.fit([X_train, XP_train], [y_train, y_train, y_train], epochs=nb_epoch,
                                batch_size=batch_size,
                                verbose=0, shuffle=True, callbacks=callbacks_list)

            toc0 = time.time()

            losses = history.history

            print('Key: ', losses.keys())

            XP_test = XP_test.reshape(-1, w, w, num_PC)

            tic1 = time.time()
            PredictLabel = model.predict([X_test, XP_test], verbose=0)[0].argmax(axis=-1)

            toc1 = time.time()

            OA, Kappa, ProducerA = CalAccuracy(PredictLabel, Y_test[:, 0])

            OAJoint[0:nb_classes, r] = ProducerA * 100
            OAJoint[-2, r] = OA * 100
            OAJoint[-1, r] = Kappa * 100

            TimeJoint[0, r] = toc0 - tic0
            TimeJoint[1, r] = toc1 - tic1

            print()
            print('rand' + repr(r + 1), 'dataID_' + repr(dataID), 'TCN&ML OA: ', OA * 100)
            print()
            print()

            XP = XP.reshape(-1, w, w, num_PC)
            Map = model.predict([X, XP], verbose=0)

            Joint = Map[0].argmax(axis=-1)

            X_result = DrawResult(Joint, dataID)

            print('\n\n\nJoint', Joint)

            plt.imsave(storage_location + repr(dataID) + '_TCN&ML_r' + repr(r + 1) + '_w' + repr(w) + 'PC' + repr(
                num_PC) + 'OA_' + repr(
                int(OA * 10000)) + '.png', X_result)

            if (r == (randtime - 1)):
                OAmean = np.mean(OAJoint[-2])
                OA_all_std = np.std(OAJoint, axis=1, ddof=0)  # ddof=0: 总体偏差
                OAstd = OA_all_std[-2]
                Kappamean = np.mean(OAJoint[-1])
                Kappastd = OA_all_std[-1]
                ProducerAmean = np.mean(OAJoint[0:nb_classes], axis=1)
                ProducerAstd = OA_all_std[0:nb_classes]
                Timemean_tra = np.mean(TimeJoint[0])
                Timestd = np.std(TimeJoint, axis=1, ddof=0)
                Timemean_tes = np.mean(TimeJoint[1])
                Timestd_tra = Timestd[0]
                Timestd_tes = Timestd[1]

                file_name = (storage_location + str(dataID) + '_Mean_TCN&ML' + '_w' + repr(w) + 'PC' + repr(
                    num_PC) + '_' + repr(int(OAmean * 10000)) + '.txt')
                with open(file_name, 'w') as x_file:
                    x_file.write('{} OA mean (%)'.format(OAmean))
                    x_file.write('\n')
                    x_file.write('\n')
                    x_file.write('{} OA std'.format(OAstd))
                    x_file.write('\n')
                    x_file.write('\n')
                    x_file.write('{} Kappa mean (%)'.format(Kappamean))
                    x_file.write('\n')
                    x_file.write('\n')
                    x_file.write('{} Kappa std'.format(Kappastd))
                    x_file.write('\n')
                    x_file.write('\n')
                    x_file.write('{} Producer Accuracy mean (%)'.format(ProducerAmean))
                    x_file.write('\n')
                    x_file.write('\n')
                    x_file.write('{} Producer Accuracy std'.format(ProducerAstd))
                    x_file.write('\n')
                    x_file.write('\n')
                    x_file.write('{} Time mean(train) '.format(Timemean_tra))
                    x_file.write('\n')
                    x_file.write('\n')
                    x_file.write('{} Time mean(train) std '.format(Timestd_tra))
                    x_file.write('\n')
                    x_file.write('\n')
                    x_file.write('{} Time mean(test) '.format(Timemean_tes))
                    x_file.write('\n')
                    x_file.write('\n')
                    x_file.write('{} Time mean(test) std '.format(Timestd_tes))
                    x_file.write('\n')
                    x_file.write('\n')
                    x_file.write('{} Array for all '.format(OAJoint))

print('**********************   Time(train) is: %6.2f' % (toc0 - tic0) + 's.   **********************\n\n')
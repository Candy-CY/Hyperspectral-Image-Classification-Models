# -*- coding: utf-8 -*-
"""
@author: mengxue.zhang
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, Dropout
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.initializers import Initializer
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from data import image_size_dict

from spatialattention import SpatialAttention
from secondpooling import SecondOrderPooling


def get_model(img_rows, img_cols, num_PC, nb_classes, dataID=1, type='aspn', lr=0.01):
    if num_PC == 0:
        num_PC = image_size_dict[str(dataID)][2]
    if type == 'spn':
        model = spn(img_rows, img_cols, num_PC, nb_classes)
    elif type == 'aspn':
        model = aspn(img_rows, img_cols, num_PC, nb_classes)
    else:
        print('invalid model type, default use aspn model')
        model = aspn(img_rows, img_cols, num_PC, nb_classes)

    rmsp = RMSprop(lr=lr, rho=0.9, epsilon=1e-05)
    model.compile(optimizer=rmsp, loss='categorical_crossentropy',
                          metrics=['accuracy'])
    return model

class Symmetry(Initializer):
    """N*N*C Symmetry initial
    """
    def __init__(self, n=200, c=16, seed=0):
        self.n = n
        self.c = c
        self.seed = seed

    def __call__(self, shape, dtype=None):
        rv = K.truncated_normal([self.n, self.n, self.c], 0., 1e-5, dtype=dtype, seed=self.seed)
        rv = (rv + K.permute_dimensions(rv, pattern=(1, 0, 2))) / 2.0
        return K.reshape(rv, [self.n * self.n, self.c])


def spn(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')

    F = Reshape([img_rows * img_cols, num_PC])(CNNInput)
    F = BatchNormalization()(F)
    F = Dropout(rate=0.5)(F)

    F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='f2')(F)
    F = SecondOrderPooling(name='feature1')(F)
    F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='feature2')(F)

    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=num_PC, c=nb_classes))(F)
    model = Model(inputs=[CNNInput], outputs=F)

    return model

def aspn(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')

    F = Reshape([img_rows * img_cols, num_PC])(CNNInput)
    F = BatchNormalization()(F)
    F = Dropout(rate=0.5)(F)

    F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='f2')(F)
    F = SpatialAttention(name='f3')(F)
    F = SecondOrderPooling(name='feature1')(F)
    F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='feature2')(F)

    n = math.ceil(math.sqrt(K.int_shape(F)[-1]))
    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=n, c=nb_classes))(F)
    model = Model(inputs=[CNNInput], outputs=F)

    return model


def get_callbacks(decay=0.0001):
    def step_decay(epoch, lr):
        return lr * math.exp(-1 * epoch * decay)

    callbacks = []
    callbacks.append(LearningRateScheduler(step_decay))

    return callbacks

from __future__ import absolute_import, division
from keras.models import Model
from keras.layers import Input, Dense, Reshape, multiply, BatchNormalization, GlobalAvgPool2D, Permute, \
    GlobalAveragePooling3D, GlobalAveragePooling2D, Flatten, Dropout
from keras.layers.convolutional import Conv3D, Conv2D
from keras import backend as K
from keras import regularizers
from Utils.layers import ConvOffset2D
from keras.utils import plot_model
import numpy as np
from sklearn.decomposition import PCA
from Utils.sepconv3D import SeparableConv3D
import keras

K.clear_session()


def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    init = input
    filters = init._keras_shape[-1]
    print("1" * 10, filters)

    se_shape = (1, 1, filters)

    # Squeeze
    se = GlobalAveragePooling2D()(init)
    print(se.shape)

    se = Reshape(se_shape)(se)

    # Excitation
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    return se


# 组合模型
class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_shape)

        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
        print('change input shape:', input_shape)

        input = Input(shape=input_shape)

        conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input)
        conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
        conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer2)
        print(conv_layer3._keras_shape)
        conv3d_shape = conv_layer3._keras_shape
        conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3] * conv3d_shape[4]))(conv_layer3)
        conv_layer4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv_layer3)

        flatten_layer = Flatten()(conv_layer4)

        ## fully connected layers
        dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
        dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer = Dense(units=num_outputs, activation='softmax')(dense_layer2)

        model = Model(inputs=input, outputs=output_layer)
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        # (1,7,7,200),16
        return ResnetBuilder.build(input_shape, num_outputs)


import tensorflow as tf


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


def main():
    model = ResnetBuilder.build_resnet_8((1, 19, 19, 200), 16)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()
    sess = K.get_session()
    graph = sess.graph
    stats_graph(graph)
    plot_model(model=model, to_file='model-deformableconv-CNN.png', show_shapes=True)


if __name__ == '__main__':
    main()

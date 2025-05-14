from __future__ import absolute_import, division
from keras.models import Model
from keras.layers import Input, Dense, Reshape, multiply, BatchNormalization, GlobalAvgPool2D, Permute, \
    GlobalAveragePooling3D, GlobalAveragePooling2D
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

        # input = applyPCA(input, 30)

        conv1 = Conv3D(filters=16, kernel_size=(3, 3, 7), strides=(1, 1, 5), kernel_regularizer=regularizers.l2(0.01),
                       kernel_initializer='he_normal', use_bias=False, activation='relu')(input)
        conv1 = BatchNormalization()(conv1)
        conv2 = Conv3D(filters=32, kernel_size=(3, 3, 5), strides=(1, 1, 3), kernel_regularizer=regularizers.l2(0.01),
                       kernel_initializer='he_normal', use_bias=False, activation='relu')(conv1)
        conv2 = BatchNormalization()(conv2)
        print(conv2.shape)
        conv3 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), kernel_regularizer=regularizers.l2(0.01),
                       kernel_initializer='he_normal', use_bias=False, activation='relu')(conv2)
        conv3 = BatchNormalization()(conv3)
        print(conv3.shape)
        conv3 = SeparableConv3D(filters=128, kernel_size=(3, 3, 1), strides=(1, 1, 1),
                                kernel_initializer=regularizers.l2(0.01),
                                use_bias=False, activation='relu')(conv3)
        print(conv3._keras_shape)
        conv3_shape = conv3._keras_shape

        l = Reshape((conv3_shape[1], conv3_shape[2], conv3_shape[3] * conv3_shape[4]))(conv3)
        print(l)

        # conv11
        l = Conv2D(32, (1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer='he_normal',
                   use_bias=False, activation='relu')(l)
        l = BatchNormalization()(l)
        print(l.shape)

        # l = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01), kernel_initializer='he_normal',
        #            use_bias=False, activation='relu')(l)
        # l3 = BatchNormalization()(l)

        # conv12
        l_offset = ConvOffset2D(32)(l)
        l1 = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=regularizers.l2(0.01),
                    kernel_initializer='he_normal', use_bias=False, activation='relu')(l_offset)
        l1 = BatchNormalization()(l1)
        print(l1.shape)

        # conv21
        l_offset = ConvOffset2D(32)(l)
        l2 = Conv2D(64, (5, 5), padding='same', strides=(1, 1), dilation_rate=5,
                    kernel_regularizer=regularizers.l2(0.01), kernel_initializer='he_normal', use_bias=False,
                    activation='relu')(l_offset)
        l2 = BatchNormalization()(l2)
        print(l2.shape)

        l = keras.layers.add([l1, l2])

        se = squeeze_excite_block(l)
        l1 = multiply([l1, se])
        l2 = multiply([l2, se])

        l = keras.layers.add([l1, l2])

        # out
        l = GlobalAvgPool2D()(l)

        # 输入分类器
        # Classifier block
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(l)

        model = Model(inputs=input, outputs=dense)
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
    model = ResnetBuilder.build_resnet_8((1, 21, 21, 103), 9)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()
    sess = K.get_session()
    graph = sess.graph
    stats_graph(graph)
    plot_model(model=model, to_file='model-deformableconv-CNN.png', show_shapes=True)


if __name__ == '__main__':
    main()

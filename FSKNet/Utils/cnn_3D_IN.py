from __future__ import absolute_import, division
from keras.models import Model, Sequential
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout,
    Reshape,
    BatchNormalization, GlobalAvgPool2D)
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    Conv3D,
    Conv2D
)
from keras import backend as K
from keras import regularizers
from Utils.layers import ConvOffset2D
from keras.utils import plot_model


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


# 组合模型
class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_shape)
        # orignal input shape: 1,7,7,200

        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
        print('change input shape:', input_shape)

        # 用keras中函数式模型API，不用序贯模型API
        # 张量流输入
        input = Input(shape=input_shape)
        # print(input.shape)
        # ( ?,7,7,200,1 )
        conv1 = Conv3D(filters=15, kernel_size=(15, 15, 24), strides=(12, 12, 12),
                       kernel_regularizer=regularizers.l2(0.01))(
            input)
        print(conv1.shape)
        # ( None, 1,1,15,15 )
        # 这一步整一个 3*3 空间的，然后在 reshape 不就好了

        l = Reshape((15, 15, 1))(conv1)
        print(l)

        # l = Conv2D(filters=64, kernel_size=(3, 3), strides=1, kernel_regularizer=regularizers.l2(0.01))(conv2_spc)
        # print(l)
        # print(l.shape)
        # ( ?,20,60,1 )

        # conv11
        l = Conv2D(32, (3, 3), padding='same', name='conv11', trainable=False)(l)
        l = Activation('relu', name='conv11_relu')(l)
        l = BatchNormalization(name='conv11_bn')(l)

        # conv12
        l_offset = ConvOffset2D(32, name='conv12_offset')(l)
        l = Conv2D(64, (3, 3), padding='same', strides=(2, 2), name='conv12', trainable=False)(l_offset)
        l = Activation('relu', name='conv12_relu')(l)
        l = BatchNormalization(name='conv12_bn')(l)

        # conv21
        l_offset = ConvOffset2D(64, name='conv21_offset')(l)
        l = Conv2D(64, (3, 3), padding='same', strides=(2, 2), name='conv21', trainable=False)(l_offset)
        l = Activation('relu', name='conv21_relu')(l)
        l = BatchNormalization(name='conv21_bn')(l)

        # conv22
        # l_offset = ConvOffset2D(64, name='conv22_offset')(l)
        # l = Conv2D(64, (3, 3), padding='same', strides=(2, 2), name='conv22', trainable=False)(l_offset)
        # l = Activation('relu', name='conv22_relu')(l)
        # l = BatchNormalization(name='conv22_bn')(l)

        # out
        l = GlobalAvgPool2D(name='avg_pool')(l)

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
    model = ResnetBuilder.build_resnet_8((1, 15, 15, 200), 16)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()
    # plot_model(model=model, to_file='model-deformableconv-CNN.png', show_shapes=True)
    sess = K.get_session()
    graph = sess.graph
    stats_graph(graph)


if __name__ == '__main__':
    main()

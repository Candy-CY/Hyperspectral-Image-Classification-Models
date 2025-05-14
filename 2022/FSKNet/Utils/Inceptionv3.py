from keras.models import Model
import tensorflow as tf
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization
)
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    Conv3D
)
from keras import backend as K
from keras import regularizers
from keras.applications import *

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

        basemodel = InceptionV3()

        # 输入分类器
        # Classifier block
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(drop1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        # (1,7,7,200),16
        return ResnetBuilder.build(input_shape, num_outputs)

def main():
    model = ResnetBuilder.build_resnet_8((1, 11, 11, 200),16)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()

if __name__ == '__main__':
    main()

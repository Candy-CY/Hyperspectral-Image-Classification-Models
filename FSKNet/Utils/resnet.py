from __future__ import print_function

import keras
from keras.layers import *
from keras.regularizers import l2
from keras.models import Model


def resnet_layer(inputs, num_filters=64, kernel_size=3, strides=1, activation='relu', batch_normalization=True,
                 conv_first=True):
    # conv-bn-relu stack builder
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(momentum=0.9)(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization(momentum=0.9)(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10, num_res_blocks=None):
    '''
    ResNet v1 model builder
    Stacks of 2 x (3 x 3) Cnv2D-BN-Relu
    Last Relu is after the shortcut connection
    At the beginning of each stage, the feature map size is halved (downsampled) by a convolutional layer with strides=2,while
    the number of filters is doubled. within each stage, the layers have the same number filters and the same number of filters
    features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16,32
    stage 2: 8x8,  64
    The number of parameters
    ResNet20: 0.27M
    ResNet32: 0.46M
    ResNet44: 0.66M
    ResNet56: 0.85M
    ResNet110:1.7M
    :param input_shape:
    :param depth:
    :param num_classes:
    :param num_res_block:
    :return:
    '''

    num_filters = 16
    if num_res_blocks is None:
        num_res_blocks = int((depth - 2) / 6)
    else:
        num_res_blocks = list(num_res_blocks)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    # stack of residual units
    for stack in range(len(num_res_blocks)):
        for res_block in range(num_res_blocks[stack]):
            strides = 1
            # first layer but not first stack
            if stack > 0 and res_block == 0:
                # downsample
                strides = 2
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            # first layer but not first stack
            if stack > 0 and res_block == 0:
                # Linear projection residual shortcut connection to match changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top
    # v1 does not use BN after last shortcut connection-Relu
    x = GlobalMaxPooling2D()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def ResNet18(input_shape, depth, num_classes=10):
    return resnet_v1(input_shape, depth, num_classes=num_classes, num_res_blocks=[2, 2, 2, 2])


def ResNet34(input_shape, depth, num_classes=10):
    return resnet_v1(input_shape, depth, num_classes=num_classes, num_res_blocks=[3, 4, 6, 3])


if __name__ == '__main__':
    model = ResNet18((224, 224, 3), 18)
    model.summary()

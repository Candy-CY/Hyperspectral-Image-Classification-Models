import keras
from keras.models import Model
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Input, Dense, Conv2D, Conv1D, Flatten, BatchNormalization, LeakyReLU, Dropout, \
    concatenate, GlobalAveragePooling2D, Activation, multiply, Permute, dot
from keras.layers.core import Reshape
from keras.layers.merge import add
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    AveragePooling2D,
    Conv2D,
    Conv3D,
    MaxPooling2D,
    MaxPooling3D
)

from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2

nb_classes = 13

num_filters_spe = 16
num_filters_spa = 16
r = 3


def model():
    input_1 = Input(shape=(7, 7, 176))
    input_2 = Input(shape=(27, 27, 30))

    CAB_conv1 = Conv2D(16, (3, 3), padding='same', strides=(1, 1),
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(1e-4),
                       #  use_bias=False
                       )(input_1)
    CAB_bn1 = BatchNormalization()(CAB_conv1)
    CAB_relu1 = PReLU()(CAB_bn1)
    CAB_avg_pool1 = AveragePooling2D()(CAB_relu1)
    CAB_conv4 = Conv2D(32, (3, 3), padding='same', strides=(1, 1),
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(1e-4),
                       #  use_bias=False
                       )(CAB_avg_pool1)
    CAB_bn4 = BatchNormalization()(CAB_conv4)
    CAB_relu4 = PReLU()(CAB_bn4)
    CAB_conv5 = Conv2D(32, (3, 3), padding='same', strides=(1, 1),
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(1e-4),
                       #  use_bias=False
                       )(CAB_relu4)
    CAB_bn5 = BatchNormalization()(CAB_conv5)
    CAB_relu5 = PReLU()(CAB_bn5)
    CAB_global_pool = GlobalAveragePooling2D()(CAB_relu5)
    # ===================================================================================================================
    CAB_reshape = Reshape((1, CAB_global_pool._keras_shape[1]))(CAB_global_pool)

    CAB_conv6 = Conv1D(44, (32), padding='same', strides=(1), kernel_initializer='glorot_uniform', use_bias=False)(
        CAB_reshape)
    CAB_bn6 = BatchNormalization()(CAB_conv6)
    CAB_relu6 = PReLU()(CAB_bn6)

    CAB_conv7 = Conv1D(176, (44), padding='same', strides=(1), kernel_initializer='glorot_uniform', use_bias=False)(
        CAB_relu6)
    CAB_bn7 = BatchNormalization()(CAB_conv7)
    CAB_sigmoid = Activation('sigmoid')(CAB_bn7)
    # ==================================================================================================================
    CAB_mul = multiply([input_1, CAB_sigmoid])

    input_spe = Reshape((CAB_mul._keras_shape[1], CAB_mul._keras_shape[2], CAB_mul._keras_shape[3], 1))(
        CAB_mul)

    # input_spe = Reshape((input_1._keras_shape[1], input_1._keras_shape[2], input_1._keras_shape[3], 1))(input_1)


    conv_spe1 = Conv3D(32, (1, 1, 7), padding='valid', strides=(1, 1, 2))(input_spe)
    print('conv_spe shape:', conv_spe1.shape)
    bn_spe1 = BatchNormalization()(conv_spe1)
    relu_spe1 = PReLU()(bn_spe1)

    conv_spe11 = Conv3D(num_filters_spe, (1, 1, 7), padding='same', strides=(1, 1, 1))(relu_spe1)
    bn_spe11 = BatchNormalization()(conv_spe11)
    relu_spe11 = PReLU()(bn_spe11)

    blockconv_spe1 = Conv3D(num_filters_spe, (1, 1, 7), padding='same', strides=(1, 1, 1))(relu_spe11)
    print('blockconv_spe1:', blockconv_spe1.shape)
    blockbn_spe1 = BatchNormalization()(blockconv_spe1)
    blockrelu_spe1 = PReLU()(blockbn_spe1)
    conv_spe2 = Conv3D(num_filters_spe, (1, 1, 7), padding='same', strides=(1, 1, 1))(blockrelu_spe1)
    print('conv_spe2 shape:', conv_spe2.shape)

    add_spe1 = add([relu_spe11, conv_spe2])

    bn_spe2 = BatchNormalization()(add_spe1)
    relu_spe2 = PReLU()(bn_spe2)

    blockconv_spe2 = Conv3D(num_filters_spe, (1, 1, 7), padding='same', strides=(1, 1, 1))(relu_spe2)
    print('blockconv_spe2 shape:', blockconv_spe2.shape)
    blockbn_spe2 = BatchNormalization()(blockconv_spe2)
    blockrelu_spe2 = PReLU()(blockbn_spe2)
    conv_spe4 = Conv3D(num_filters_spe, (1, 1, 7), padding='same', strides=(1, 1, 1))(blockrelu_spe2)
    print('conv_spe_4 shape:', conv_spe4.shape)

    add_spe2 = add([relu_spe2, conv_spe4])

    bn_spe4 = BatchNormalization()(add_spe2)
    relu_spe4 = PReLU()(bn_spe4)

    blockconv_spe3 = Conv3D(num_filters_spe, (1, 1, 7), padding='same', strides=(1, 1, 1))(relu_spe4)
    blockbn_spe3 = BatchNormalization()(blockconv_spe3)
    blockrelu_spe3 = PReLU()(blockbn_spe3)

    conv_spe41 = Conv3D(num_filters_spe, (1, 1, 7), padding='same', strides=(1, 1, 1))(blockrelu_spe3)

    add_spe3 = add([relu_spe4, conv_spe41])
    # ===================================================================================================

    bn_spe41 = BatchNormalization()(add_spe3)
    relu_spe41 = PReLU()(bn_spe41)


    add_all_spe = add([relu_spe2, relu_spe4, relu_spe41])

    conv_spe6 = Conv3D(8, (1, 1, 85), padding='valid', strides=(1, 1, 1))(add_all_spe)
    print('conv_spe_3 shape:', conv_spe6.shape)
    bn_spe6 = BatchNormalization()(conv_spe6)
    relu_spe6 = PReLU()(bn_spe6)


    input_spa = Reshape((input_2._keras_shape[1], input_2._keras_shape[2], input_2._keras_shape[3], 1))(input_2)

    conv_spa1 = Conv3D(16, (5, 5, 30), padding='valid', strides=(1, 1, 1))(input_spa)
    print('conv_spa1 shape:', conv_spa1.shape)
    bn_spa1 = BatchNormalization()(conv_spa1)
    relu_spa1 = PReLU()(bn_spa1)
    reshape_spa1 = Reshape((relu_spa1._keras_shape[1], relu_spa1._keras_shape[2], relu_spa1._keras_shape[4], relu_spa1._keras_shape[3]))(relu_spa1)


    conv_spa11 = Conv3D(num_filters_spa, (3, 3, 1), padding='same', strides=(1, 1, 1))(reshape_spa1)
    bn_spa11 = BatchNormalization()(conv_spa11)
    relu_spa11 = PReLU()(bn_spa11)

    VIS_conv1 = Conv3D(16, (1, 1, 16), padding='valid', strides=(1, 1, 1),kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(1e-4))(relu_spa11)
    VIS_BN1 = BatchNormalization()(VIS_conv1)

    VIS_relu1 = Activation('relu')(VIS_BN1)
    VIS_SHAPE1 = Reshape((VIS_relu1._keras_shape[1] * VIS_relu1._keras_shape[2], VIS_relu1._keras_shape[4]))(
        VIS_relu1)

    VIS_conv2 = Conv3D(16, (1, 1, 16), padding='valid', strides=(1, 1, 1),kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(1e-4))(relu_spa11)
    VIS_BN2 = BatchNormalization()(VIS_conv2)
    VIS_relu2 = Activation('relu')(VIS_BN2)
    VIS_SHAPE2 = Reshape((VIS_relu2._keras_shape[1] * VIS_relu2._keras_shape[2], VIS_relu2._keras_shape[4]))(
        VIS_relu2)
    trans_VIS_SHAPE2 = Permute((2, 1))(VIS_SHAPE2)

    VIS_conv3 = Conv3D(16, (1, 1, 16), padding='valid', strides=(1, 1, 1),kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(1e-4))(relu_spa11)
    VIS_BN3 = BatchNormalization()(VIS_conv3)
    VIS_relu3 = Activation('relu')(VIS_BN3)
    VIS_SHAPE3 = Reshape((VIS_relu3._keras_shape[1] * VIS_relu3._keras_shape[2], VIS_relu3._keras_shape[4]))(
        VIS_relu3)

    VIS_mul1 = dot([VIS_SHAPE1, trans_VIS_SHAPE2], axes=(2, 1))

    VIS_sigmoid = Activation('sigmoid')(VIS_mul1)

    VIS_mul2 = dot([VIS_sigmoid, VIS_SHAPE3], axes=(2, 1))

    VIS_SHAPEall = Reshape((23,23,16, 1))(VIS_mul2)

    VIS_conv4 = Conv3D(16, (16, 1, 1), padding='same', strides=(1),kernel_initializer='glorot_uniform',
                       kernel_regularizer=l2(1e-4))(VIS_SHAPEall)
    VIS_BN4 = BatchNormalization()(VIS_conv4)
    VIS_ADD = add([relu_spa11, VIS_BN4])


    blockconv_spa1 = Conv3D(num_filters_spa, (3, 3, 1), padding='same', strides=(1, 1, 1))(VIS_ADD)
    blockbn_spa1 = BatchNormalization()(blockconv_spa1)
    blockrelu_spa1 = PReLU()(blockbn_spa1)
    conv_spa2 = Conv3D(num_filters_spa, (3, 3, 1), padding='same', strides=(1))(blockrelu_spa1)
    print('conv_spa_2 shape:', conv_spa2.shape)

    add_spa1 = add([VIS_ADD, conv_spa2])

    bn_spa2 = BatchNormalization()(add_spa1)
    relu_spa2 = PReLU()(bn_spa2)


    blockconv_spa2 = Conv3D(num_filters_spa, (3, 3, 1), padding='same', strides=(1))(relu_spa2)
    print('blockconv_spa12', blockconv_spa2.shape)
    blockbn_spa2 = BatchNormalization()(blockconv_spa2)
    blockrelu_spa2 = PReLU()(blockbn_spa2)
    conv_spa4 = Conv3D(num_filters_spa, (3, 3, 1), padding='same', strides=(1))(blockrelu_spa2)
    print('conv_spa4 shape:', conv_spa4.shape)

    add_spa2 = add([relu_spa2, conv_spa4])
    bn_spa4 = BatchNormalization()(add_spa2)
    relu_spa4 = PReLU()(bn_spa4)


    blockconv_spa3 = Conv3D(num_filters_spa, (3, 3, 1), padding='same', strides=(1))(relu_spa4)
    blockbn_spa3 = BatchNormalization()(blockconv_spa3)
    blockrelu_spa3 = PReLU()(blockbn_spa3)
    conv_spa41 = Conv3D(num_filters_spa, (3, 3, 1), padding='same', strides=(1))(blockrelu_spa3)
    add_spa3 = add([relu_spa4, conv_spa41])
    bn_spa41 = BatchNormalization()(add_spa3)
    relu_spa41 = PReLU()(bn_spa41)

    add_all_spa = add([relu_spa2, relu_spa4, relu_spa41])


    conv_spa6 = Conv3D(num_filters_spa, (5, 5, 1), padding='valid', strides=(1, 1, 1))(add_all_spa)
    bn_spa6 = BatchNormalization()(conv_spa6)
    relu_spa6 = PReLU()(bn_spa6)


    conv_spa7 = Conv3D(num_filters_spa, (5, 5, 1), padding='valid', strides=(1, 1, 1))(relu_spa6)
    bn_spa7 = BatchNormalization()(conv_spa7)
    relu_spa7 = PReLU()(bn_spa7)

    conv_spa8 = Conv3D(num_filters_spa, (5, 5, 1), padding='valid', strides=(1, 1, 1))(relu_spa7)
    bn_spa8 = BatchNormalization()(conv_spa8)
    relu_spa8 = PReLU()(bn_spa8)

    conv_spa81 = Conv3D(num_filters_spa, (5, 5, 1), padding='valid', strides=(1, 1, 1))(relu_spa8)
    bn_spa81 = BatchNormalization()(conv_spa81)
    relu_spa81 = PReLU()(bn_spa81)

    conv_spa9 = Conv3D(8, (1, 1, 16), padding='valid', strides=(1, 1, 1))(relu_spa81)
    bn_spa9 = BatchNormalization()(conv_spa9)
    relu_spa9 = PReLU()(bn_spa9)


    feature_fusion = concatenate([relu_spe6, relu_spa9])
    reshape_all = Reshape((feature_fusion._keras_shape[1], feature_fusion._keras_shape[2],
                           feature_fusion._keras_shape[4], feature_fusion._keras_shape[3]))(
        feature_fusion)

    conv_all1 = Conv3D(16, (3), padding='same', strides=(1, 1, 1))(reshape_all)
    print('convall1 shape:', conv_all1.shape)
    bn_all1 = BatchNormalization()(conv_all1)
    relu_all1 = PReLU()(bn_all1)

    VIS_conv11 = Conv3D(16, (1, 1, 16), padding='valid', strides=(1, 1, 1), kernel_initializer='glorot_uniform',
                        kernel_regularizer=l2(1e-4))(relu_all1)
    VIS_BN11 = BatchNormalization()(VIS_conv11)
    VIS_relu11 = Activation('relu')(VIS_BN11)
    VIS_SHAPE11 = Reshape((VIS_relu11._keras_shape[1] * VIS_relu11._keras_shape[2], VIS_relu11._keras_shape[4]))(
        VIS_relu11)

    VIS_conv21 = Conv3D(16, (1, 1, 16), padding='valid', strides=(1, 1, 1), kernel_initializer='glorot_uniform',
                        kernel_regularizer=l2(1e-4))(relu_all1)
    VIS_BN21 = BatchNormalization()(VIS_conv21)
    VIS_relu21 = Activation('relu')(VIS_BN21)
    VIS_SHAPE21 = Reshape((VIS_relu21._keras_shape[1] * VIS_relu21._keras_shape[2], VIS_relu21._keras_shape[4]))(
        VIS_relu21)
    trans_VIS_SHAPE21 = Permute((2, 1))(VIS_SHAPE21)

    VIS_conv31 = Conv3D(16, (1, 1, 16), padding='valid', strides=(1, 1, 1), kernel_initializer='glorot_uniform',
                        kernel_regularizer=l2(1e-4))(relu_all1)
    VIS_BN31 = BatchNormalization()(VIS_conv31)
    VIS_relu31 = Activation('relu')(VIS_BN31)
    VIS_SHAPE31 = Reshape((VIS_relu31._keras_shape[1] * VIS_relu31._keras_shape[2], VIS_relu31._keras_shape[4]))(
        VIS_relu31)

    VIS_mul11 = dot([VIS_SHAPE11, trans_VIS_SHAPE21], axes=(2, 1))

    VIS_sigmoid1 = Activation('sigmoid')(VIS_mul11)

    VIS_mul21 = dot([VIS_sigmoid1, VIS_SHAPE31], axes=(2, 1))

    VIS_SHAPEall1 = Reshape((7, 7, 16, 1))(VIS_mul21)

    VIS_conv41 = Conv3D(16, (16, 1, 1), padding='same', strides=(1), kernel_initializer='glorot_uniform',
                        kernel_regularizer=l2(1e-4))(VIS_SHAPEall1)
    VIS_BN41 = BatchNormalization()(VIS_conv41)
    VIS_ADD1 = add([relu_all1, VIS_BN41])

    conv_all2 = Conv3D(16, (3), padding='valid', strides=(1, 1, 1))(VIS_ADD1)
    bn_all2 = BatchNormalization()(conv_all2)
    relu_all2 = PReLU()(bn_all2)

    flatten = Flatten()(relu_all2)
    dense = Dense(units=512, activation="relu", kernel_initializer="he_normal")(flatten)
    drop = Dropout(0.6)(dense)
    dense_2 = Dense(units=256, activation="relu", kernel_initializer="he_normal")(drop)
    drop1 = Dropout(0.6)(dense_2)
    dense_3 = Dense(units=nb_classes, activation="softmax", kernel_initializer="he_normal")(drop1)

    model = Model(inputs=[input_1, input_2], outputs=dense_3)
    sgd = SGD(lr=0.0005, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd , metrics=['accuracy'])
    model.summary()
    return model

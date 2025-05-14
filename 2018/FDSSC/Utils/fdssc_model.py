from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    AveragePooling3D,
    Conv3D
)
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU, ThresholdedReLU
from keras.regularizers import l2
from keras import backend as K
from keras.layers.core import Reshape
from keras import regularizers
from keras.layers.merge import concatenate


def bn_prelu(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    # return Activation("relu")(norm)
    return PReLU()(norm)


def spectral_conv(input):
    activation = bn_prelu(input)
    conv = Conv3D(kernel_initializer='he_normal', strides=(1, 1, 1), kernel_regularizer=l2(0.0001),
                  filters=growth_rate, kernel_size=(1, 1, 7), padding='same', dilation_rate=(1, 1, 1))(activation)
    return conv


def spatial_conv(input):
    activation = bn_prelu(input)
    conv = Conv3D(kernel_initializer='he_normal', strides=(1, 1, 1), kernel_regularizer=l2(0.0001),
                  filters=growth_rate, kernel_size=(3, 3, 1), padding='same', dilation_rate=(1, 1, 1))(activation)
    return conv


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


class fdssc_model(object):
    @staticmethod
    def build(input_shape, num_outputs):
        global growth_rate
        growth_rate = 12
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])

        input = Input(shape=input_shape)
        print("the dim of input:", input._keras_shape[3])
        # Dense spectral block

        x1_0 = Conv3D(kernel_initializer='he_normal', strides=(1, 1, 2), kernel_regularizer=regularizers.l2(0.0001),
                      filters=24, kernel_size=(1, 1, 7), padding='valid')(input)
        x1_1 = spectral_conv(x1_0)
        x1_1_ = concatenate([x1_0, x1_1], axis=CHANNEL_AXIS)
        x1_2 = spectral_conv(x1_1_)
        x1_2_ = concatenate([x1_0, x1_1, x1_2], axis=CHANNEL_AXIS)
        x1_3 = spectral_conv(x1_2_)
        x1 = concatenate([x1_0, x1_1, x1_2, x1_3], axis=CHANNEL_AXIS)
        x1 = bn_prelu(x1)

        print('the output of dense spectral block:', x1._keras_shape)

        # Reducing dimension layer
        # x1 = Conv3D(kernel_initializer='he_normal', strides=(1, 1, 1), kernel_regularizer=regularizers.l2(0.0001),
        #            filters=24, kernel_size=(1, 1, 1), padding='valid')(x1)
        tran1 = Conv3D(kernel_initializer='he_normal', strides=(1, 1, 1), kernel_regularizer=regularizers.l2(0.0001),
                       filters=200, kernel_size=(1, 1, x1._keras_shape[CONV_DIM3]), padding='valid')(x1)
        print(tran1._keras_shape)
        tran1 = bn_prelu(tran1)
        tran2 = Reshape((tran1._keras_shape[CONV_DIM1], tran1._keras_shape[CONV_DIM2],
                         tran1._keras_shape[CHANNEL_AXIS], 1))(tran1)

        x2_0 = Conv3D(kernel_initializer='he_normal', strides=(1, 1, 1), kernel_regularizer=regularizers.l2(0.0001),
                      filters=24, kernel_size=(3, 3, 200), padding='valid')(tran2)
        print('the input of dense spatial block:', x2_0._keras_shape)

        # Dense spatial block
        x2_1 = spatial_conv(x2_0)
        x2_1_ = concatenate([x2_0, x2_1], axis=CHANNEL_AXIS)
        x2_2 = spatial_conv(x2_1_)
        x2_2_ = concatenate([x2_0, x2_1, x2_2], axis=CHANNEL_AXIS)
        x2_3 = spatial_conv(x2_2_)
        x2 = concatenate([x2_0, x2_1, x2_2, x2_3], axis=CHANNEL_AXIS)

        print('the output of dense spectral block is:', x2._keras_shape)
        x2 = bn_prelu(x2)

        # Classifier block
        pool1 = AveragePooling3D(pool_size=(x2._keras_shape[1], x2._keras_shape[2], 1), strides=(1, 1, 1))(x2)

        flatten1 = Flatten()(pool1)
        drop1 = Dropout(0.5)(flatten1)
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="glorot_normal")(drop1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_fdssc(input_shape, num_outputs):
        return fdssc_model.build(input_shape, num_outputs)


def main():
    model = fdssc_model.build_fdssc((1, 9, 9, 176), 13)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()


if __name__ == '__main__':
    main()

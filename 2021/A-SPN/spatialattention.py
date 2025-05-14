# -*- coding: utf-8 -*-
"""
@author: mengxue.zhang
"""

from keras.layers import Layer
import keras.backend as K


class SpatialAttention(Layer):
    def __init__(self,
                 **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[input_shape[-2], 1],
                                 name='kernel',
                                 initializer='ones',
                                 trainable=True)

        self.bias = self.add_weight(shape=[input_shape[-2]],
                                 name='bias',
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs, training=None):

        input_shape = K.int_shape(inputs)
        mid = input_shape[-2] // 2

        coe = K.l2_normalize(K.batch_dot(inputs, K.permute_dimensions(inputs, pattern=(0, 2, 1))), axis=-1)
        coe0 = K.expand_dims(coe[:, mid, :], axis=-1) * self.kernel
        w = K.batch_dot(coe, coe0) + K.expand_dims(self.bias, axis=-1)
        outputs = K.softmax(w, axis=-2) * inputs

        return outputs

    def get_config(self):
        config = {}
        base_config = super(SpatialAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)


# -*- coding: utf-8 -*-
"""
@author: Snow
"""

from keras.layers import Layer
import keras.backend as K


class SecondOrderPooling(Layer):
    def __init__(self,
                 **kwargs):
        super(SecondOrderPooling, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)

        outputs = K.batch_dot(K.permute_dimensions(inputs, pattern=(0, 2, 1)), inputs, axes=[2, 1])
        outputs = K.reshape(outputs, [-1, input_shape[2] * input_shape[2]])

        return outputs

    def get_config(self):
        config = {}
        base_config = super(SecondOrderPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 4:
            output_shape = list([None, input_shape[1], input_shape[3] * input_shape[3]])
        else:
            output_shape = list([None, input_shape[2] * input_shape[2]])
        return tuple(output_shape)


"""
@author: mqalkhatib
"""

from tensorflow.keras.layers import Conv3D, Conv2D, Conv1D
from tensorflow.keras.layers import Dense, Flatten, Reshape, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout,Input
import tensorflow as tf
from cvnn.layers import ComplexConv3D, complex_input, ComplexFlatten, ComplexDense, ComplexDropout, ComplexConv2D

def dual_Path(x, x_fft, num_class):

    # standard Path
    input_layer1 = Input(shape=x.shape[1:], name = "x")
     ## convolutional layers
    conv_layer1 = Conv3D(filters = 32, kernel_size=(3, 3, 3), activation= 'relu')(input_layer1)
    conv_layer2 = Conv3D(filters = 32, kernel_size=(3, 3, 3), activation= 'relu')(conv_layer1)
    conv_layer3 = Conv3D(filters = 32, kernel_size=(3, 3, 3), activation= 'relu')(conv_layer2)

    flatten_layer = ComplexFlatten()(tf.dtypes.cast(conv_layer3, tf.complex64))
    #flatten_layer = Flatten()(conv_layer3)
    
    # FFT Path
    fft_inputs = complex_input(shape=x_fft.shape[1:], name= "x_fft")
    c1 = ComplexConv3D(filters = 32, kernel_size=(3,3,3), activation='cart_relu')(fft_inputs)
    c2 = ComplexConv3D(filters = 32, kernel_size=(3,3,3), activation='cart_relu' )(c1)
    c3 = ComplexConv3D(filters = 32, kernel_size=(3,3,3), activation='cart_relu')(c2)

    cmplx_flatten_layer = ComplexFlatten()(c3)

    result = tf.concat([flatten_layer,cmplx_flatten_layer],axis=1);



    ## fully connected layers
    dense_layer1 = ComplexDense(units = 128, activation='cart_relu')(result)
    dense_layer1 = ComplexDropout(0.3)(dense_layer1)
    dense_layer2 = ComplexDense(units = 64, activation='cart_relu')(dense_layer1)
    dense_layer2 = ComplexDropout(0.2)(dense_layer2)
    output_layer = ComplexDense(num_class,activation="softmax_real_with_abs")(dense_layer2)
    
    
    model=Model(inputs=[input_layer1,fft_inputs],outputs=output_layer)
    model.compile(optimizer='ADAM',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model
    


def dual_Path_with_SE(x, x_fft, num_class):

    # standard Path
    input_layer1 = Input(shape=x.shape[1:], name = "x")
     ## convolutional layers
    conv_layer1 = Conv3D(filters = 32, kernel_size=(3, 3, 3), activation= 'relu')(input_layer1)
    conv_layer2 = Conv3D(filters = 32, kernel_size=(3, 3, 3), activation= 'relu')(conv_layer1)
    conv_layer3 = Conv3D(filters = 32, kernel_size=(3, 3, 3), activation= 'relu')(conv_layer2)
    
    # convert features into complex datatype
    realToCmplx = tf.dtypes.cast(conv_layer3, tf.complex64)
    
    # FFT Path
    fft_inputs = complex_input(shape=x_fft.shape[1:], name= "x_fft")
    c1 = ComplexConv3D(filters = 32, kernel_size=(3,3,3), activation='cart_relu')(fft_inputs)
    c2 = ComplexConv3D(filters = 32, kernel_size=(3,3,3), activation='cart_relu' )(c1)
    c3 = ComplexConv3D(filters = 32, kernel_size=(3,3,3), activation='cart_relu')(c2)


    # Concatenate features 
    features_concat = tf.concat([realToCmplx,c3],axis=4);
    
    # Apply Attention
    se = cmplx_SE_Block(features_concat, se_ratio = 8)
    se = cmplx_SE_Block(se, se_ratio = 8)
    se = cmplx_SE_Block(se, se_ratio = 8)
    se = cmplx_SE_Block(se, se_ratio = 8)
    se = cmplx_SE_Block(se, se_ratio = 8)
    se = cmplx_SE_Block(se, se_ratio = 8)
    se = cmplx_SE_Block(se, se_ratio = 8)




    cmplx_flatten_layer = ComplexFlatten()(se)


    ## fully connected layers
    dense_layer1 = ComplexDense(units = 128, activation='cart_relu')(cmplx_flatten_layer)
    dense_layer1 = ComplexDropout(0.3)(dense_layer1)
    dense_layer2 = ComplexDense(units = 64, activation='cart_relu')(dense_layer1)
    dense_layer2 = ComplexDropout(0.2)(dense_layer2)
    output_layer = ComplexDense(num_class,activation="softmax_real_with_abs")(dense_layer2)
    
    
    model=Model(inputs=[input_layer1,fft_inputs],outputs=output_layer)
    model.compile(optimizer='ADAM',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model



import cvnn.layers as complex_layers
def cmplx_SE_Block(xin, se_ratio = 8):
    # Squeeze Path
    xin = tf.transpose(xin, perm=[0, 1, 2, 4, 3])
    xin_gap =  GlobalCmplxAveragePooling3D(xin)
    sqz = complex_layers.ComplexDense(xin.shape[-1]//se_ratio, activation='cart_relu')(xin_gap)
    
    # Excitation Path
    excite1 = complex_layers.ComplexDense(xin.shape[-1], activation='cart_sigmoid')(sqz)
    
    out = tf.keras.layers.multiply([xin, excite1])
    out = tf.transpose(out, perm=[0, 1, 2, 4, 3])
    return out
    
   

def GlobalCmplxAveragePooling3D(inputs):
    inputs_r = tf.math.real(inputs)
    inputs_i = tf.math.imag(inputs)
    
    output_r = tf.keras.layers.GlobalAveragePooling3D()(inputs_r)
    output_i = tf.keras.layers.GlobalAveragePooling3D()(inputs_i)
    
    if inputs.dtype == 'complex' or inputs.dtype == 'complex64' or inputs.dtype == 'complex128':
           output = tf.complex(output_r, output_i)
    else:
           output = output_r
    
    return output

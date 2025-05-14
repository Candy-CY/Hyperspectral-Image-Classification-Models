from keras import backend as K
from keras.engine.topology import Layer

def GELU(x):
     import tensorflow as tf
     x = x*0.5* (1.0 + tf.erf(x/tf.sqrt(2.0)))
     return x
 
class FeedForward(Layer):
    def __init__(self, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        
        
    def build(self, input_shape):
        self.inner_dim = input_shape[-1]//2
        
        self.W1 = self.add_weight(name        = 'W1',
                                  shape       = (input_shape[-1], self.inner_dim),
                                  initializer = 'glorot_uniform',
                                  trainable   = True)
        self.W2 = self.add_weight(name        = 'W2',
                                  shape       = (self.inner_dim, input_shape[-1]),
                                  initializer = 'glorot_uniform',
                                  trainable   = True)
        
        self.b1 = self.add_weight(name        = 'b1',
                                  shape       = (self.inner_dim,),
                                  initializer = 'glorot_uniform',
                                  trainable   = True)
        self.b2 = self.add_weight(name        = 'b2',
                                  shape       = (input_shape[-1],),
                                  initializer = 'glorot_uniform',
                                  trainable   = True)
        
        super(FeedForward, self).build(input_shape)     
              
    def call(self, x):
        
        # 第1层变换
        F    = x
        F    = K.dot(F,self.W1)+self.b1
        F    = GELU(F)
        # 第2层变换
        F    = K.dot(F,self.W2)+self.b2
        return F
 
    def compute_output_shape(self, input_shape):
        return input_shape




    
        
        
        
        
        
        
        
        
        
        
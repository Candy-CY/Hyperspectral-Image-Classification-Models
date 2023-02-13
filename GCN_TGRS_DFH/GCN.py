
"""
@author: danfeng
"""
#import library
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as scio 
import scipy.io as sio
from tf_utils import random_mini_batches, convert_to_one_hot
from tensorflow.python.framework import ops

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def create_placeholders(n_x, n_y):
    #tf.placeholder()函数作为一种占位符用于定义过程，可以理解为形参，在执行的时候再赋具体的值，仅仅是占位符而已。
    isTraining = tf.placeholder_with_default(True, shape=())
    x_in = tf.placeholder(tf.float32,  [None, n_x], name = "x_in")
    y_in = tf.placeholder(tf.float32, [None, n_y], name = "y_in")
    mask_train = tf.placeholder(tf.float32, name = "mask_train")
    mask_test = tf.placeholder(tf.float32, name = "mask_test")
    lap = tf.placeholder(tf.float32, [None, None], name = "lap")
    
    return x_in, y_in, lap, mask_train, mask_test, isTraining

def initialize_parameters():
    
    tf.set_random_seed(1)
    #如果变量存在，函数tf.get_variable()会返回现有的变量；如果变量不存在，会根据给定形状和初始值创建一个新的变量。
    x_w1 = tf.get_variable("x_w1", [200,128], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    x_b1 = tf.get_variable("x_b1", [128], initializer = tf.zeros_initializer())
    x_w2 = tf.get_variable("x_w2", [128,16], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    x_b2 = tf.get_variable("x_b2", [16], initializer = tf.zeros_initializer())
    
    parameters = {"x_w1": x_w1,
                  "x_b1": x_b1,
                  "x_w2": x_w2,
                  "x_b2": x_b2}
                  
    return parameters

def GCN_layer(x_in, L_, weights):

    x_mid = tf.matmul(x_in, weights)
    x_out = tf.matmul(L_, x_mid)
    
    return x_out

def mynetwork(x, parameters, Lap, isTraining, momentums = 0.9):

    with tf.name_scope("x_layer_1"):
        
         x_z1_bn = tf.layers.batch_normalization(x, momentum = momentums, training = isTraining)          
         x_z1 = GCN_layer(x_z1_bn, Lap, parameters['x_w1']) + parameters['x_b1']
         x_z1_bn = tf.layers.batch_normalization(x_z1, momentum = momentums, training = isTraining)   
         x_a1 = tf.nn.relu(x_z1_bn)     

    with tf.name_scope("x_layer_2"):
         
         x_z2_bn = tf.layers.batch_normalization(x_a1, momentum = momentums, training = isTraining)            
         x_z2 = GCN_layer(x_z2_bn, Lap, parameters['x_w2']) + parameters['x_b2']         
         
    l2_loss =  tf.nn.l2_loss(parameters['x_w1']) + tf.nn.l2_loss(parameters['x_w2'])
                
    return x_z2, l2_loss

def mynetwork_optimaization(y_est, y_re, l2_loss, mask, reg, learning_rate, global_step):
    
    with tf.name_scope("cost"):
         cost = (tf.nn.softmax_cross_entropy_with_logits(logits = y_est, labels = y_re)) +  reg * l2_loss
         mask = tf.cast(mask, dtype = tf.float32)
         mask /= tf.reduce_mean(mask)
         cost *= mask
         cost = tf.reduce_mean(cost) +  reg * l2_loss
         
    with tf.name_scope("optimization"):
         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost,  global_step=global_step)
         optimizer = tf.group([optimizer, update_ops])
         
    return cost, optimizer

def masked_accuracy(preds, labels, mask):

      correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
      accuracy = tf.cast(correct_prediction, "float")
      mask = tf.cast(mask, dtype = tf.float32)
      mask /= tf.reduce_mean(mask)
      accuracy *= mask
      
      return tf.reduce_mean(accuracy)

def train_mynetwork(x_all, y_all, L_all, mask_in, mask_out, learning_rate = 0.001, beta_reg = 0.001, num_epochs = 200, print_cost = True):
    
    ops.reset_default_graph()                                                         
    (m, n_x) = x_all.shape
    (m, n_y) = y_all.shape
    
    costs = []                                        
    costs_dev = []
    train_acc = []
    val_acc = []
    
    x_in, y_in, lap, mask_train, mask_test, isTraining = create_placeholders(n_x, n_y) 

    parameters = initialize_parameters()
    
    with tf.name_scope("network"):
         x_out, l2_loss = mynetwork(x_in, parameters, lap, isTraining)

    global_step = tf.Variable(0, trainable=False)
    
    with tf.name_scope("optimization"):
         cost, optimizer = mynetwork_optimaization(x_out, y_in, l2_loss, mask_train, beta_reg, learning_rate, global_step)

    with tf.name_scope("metrics"):
         accuracy_train = masked_accuracy(x_out, y_in, mask_train)
         accuracy_test= masked_accuracy(x_out, y_in, mask_test)
         
    init = tf.global_variables_initializer()
   
    with tf.Session() as sess:
        
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs + 1):

            _, epoch_cost, epoch_acc = sess.run([optimizer, cost, accuracy_train], feed_dict={x_in: x_all, y_in: y_all, lap: L_all, mask_train: mask_in, mask_test: mask_out, isTraining: True})
            
            if print_cost == True and epoch % 50 == 0:
                features, overall_cost_dev, overall_acc_dev = sess.run([x_out, cost, accuracy_test], feed_dict={x_in: x_all, y_in: y_all, lap: L_all, mask_train: mask_in, mask_test: mask_out, isTraining: False})
                print ("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (epoch, epoch_cost, overall_cost_dev, epoch_acc, overall_acc_dev))
            
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                train_acc.append(epoch_acc)
                costs_dev.append(overall_cost_dev)
                val_acc.append(overall_acc_dev)
      
        # plot the cost      
        plt.plot(np.squeeze(costs))
        plt.plot(np.squeeze(costs_dev))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        # plot the accuracy 
        plt.plot(np.squeeze(train_acc))
        plt.plot(np.squeeze(val_acc))
        plt.ylabel('accuracy')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
    
        return parameters , val_acc, features


ALL_X = scio.loadmat('HSI_GCN/ALL_X.mat')
ALL_Y = scio.loadmat('HSI_GCN/ALL_Y.mat')
ALL_L = scio.loadmat('HSI_GCN/ALL_L.mat')

ALL_L = ALL_L['ALL_L']
ALL_X = ALL_X['ALL_X']
ALL_Y = ALL_Y['ALL_Y']

GCN_mask_TR = sample_mask(np.arange(0,695), ALL_Y.shape[0])
GCN_mask_TE = sample_mask(np.arange(696,10366), ALL_Y.shape[0])

ALL_Y = convert_to_one_hot(ALL_Y - 1, 16)
ALL_Y = ALL_Y.T


parameters, val_acc, features = train_mynetwork(ALL_X, ALL_Y, ALL_L.todense(), GCN_mask_TR, GCN_mask_TE)
sio.savemat('features.mat', {'features': features})
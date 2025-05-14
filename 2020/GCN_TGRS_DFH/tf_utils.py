import numpy as np
import tensorflow as tf
import math

def random_mini_batches_GCN(X, Y, L, mini_batch_size, seed):
    
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    shuffled_L1 = L[permutation, :].reshape((L.shape[0], L.shape[1]), order = "F")
    shuffled_L = shuffled_L1[:, permutation].reshape((L.shape[0], L.shape[1]), order = "F")

    num_complete_minibatches = math.floor(m / mini_batch_size)
    
    for k in range(0, num_complete_minibatches):       
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_L = shuffled_L[k * mini_batch_size : k * mini_batch_size + mini_batch_size, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_L)
        mini_batches.append(mini_batch)
    mini_batch = (X, Y, L) 
    mini_batches.append(mini_batch)
    
    return mini_batches

def random_mini_batches_GCN1(X, X1, Y, L, mini_batch_size, seed):
    
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))#随机排列，np.random.permutation 得到乱序结果
    shuffled_X = X[permutation, :]
    shuffled_X1 = X1[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    shuffled_L1 = L[permutation, :].reshape((L.shape[0], L.shape[1]), order = "F")
    shuffled_L = shuffled_L1[:, permutation].reshape((L.shape[0], L.shape[1]), order = "F")

    num_complete_minibatches = math.floor(m / mini_batch_size)
    
    for k in range(0, num_complete_minibatches):       
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_X1 = shuffled_X1[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_L = shuffled_L[k * mini_batch_size : k * mini_batch_size + mini_batch_size, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_X1, mini_batch_Y, mini_batch_L)
        mini_batches.append(mini_batch)
    mini_batch = (X, X1, Y, L) 
    mini_batches.append(mini_batch)
    
    return mini_batches
        
def random_mini_batches(X1, X2, Y, mini_batch_size, seed):
    
    m = X1.shape[0]
    m1 = X2.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X1 = X1[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    
    permutation1 = list(np.random.permutation(m1))
    shuffled_X2 = X2[permutation1, :]
    
    num_complete_minibatches = math.floor(m1/mini_batch_size)
    
    mini_batch_X1 = shuffled_X1
    mini_batch_Y = shuffled_Y
      
    for k in range(0, num_complete_minibatches):        
        mini_batch_X2 = shuffled_X2[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]        
        mini_batch = (mini_batch_X1, mini_batch_X2, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def random_mini_batches_single(X1, Y, mini_batch_size, seed):
    
    m = X1.shape[0]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X1 = X1[permutation, :]
    #shuffled_X2 = X2[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    
    num_complete_minibatches = math.floor(m/mini_batch_size)
        
    for k in range(0, num_complete_minibatches):
        mini_batch_X1 = shuffled_X1[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X1, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

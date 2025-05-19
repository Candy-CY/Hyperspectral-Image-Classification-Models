import numpy as np
import random
import os
import torch
from sklearn.decomposition import PCA

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute(A,AA,acc,k,CLASS_NUM,time1,time2):
    A = np.mean(A, 1)
    AAMean = np.mean(AA,0)
    AAStd = np.std(AA)
    AMean = np.mean(A, 0)
    AStd = np.std(A, 0)
    OAMean = np.mean(acc)
    OAStd = np.std(acc)
    kMean = np.mean(k)
    kStd = np.std(k)
    print ("train time per DataSet(s): " + "{:.5f}".format(time1))
    print("test time per DataSet(s): " + "{:.5f}".format(time2))
    print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
    print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
    print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
    print ("accuracy for each class: ")
    for i in range(CLASS_NUM):
        print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

    best_iDataset = 0
    for i in range(len(acc)):
        print('{}:{}'.format(i, acc[i]))
        if acc[i] > acc[best_iDataset]:
            best_iDataset = i
    print('best acc all={}'.format(acc[best_iDataset]))

def pca(data,n):
    pca = PCA(n_components=n)   
    height, width, channels = data.shape
    data = data.reshape(-1,channels)
    data_PCA = pca.fit_transform(data).reshape(height,width,n)
    Score = pca.explained_variance_ratio_
    min_value = np.min(data_PCA, axis=(0, 1))  
    max_value = np.max(data_PCA, axis=(0, 1))  
    data_PCA = (data_PCA - min_value) / (max_value - min_value)
    return data_PCA,Score

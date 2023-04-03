'''
read HSI dataset, split training, validation, and test dataset
'''
import os
import sys
import time
import glob
import numpy as np
import torch
import h5py
import darts.utils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.io as sio
from torch.autograd import Variable
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
import scipy.io as sio
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser("HSI")

parser.add_argument('--data_root', type=str, default='data/h5')
args = parser.parse_args()


def load_data(image_file, label_file,windowsize,dataset):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    if dataset=='houston':
        image = image_data['hsi']   
        label = label_data['houston_gt_sum']  
    if dataset=='ksc':
        image = image_data['KSC']   
        label = label_data['KSC_gt']  
    if dataset=='pu':
        image = image_data['paviaU']   
        label = label_data['paviaU_gt']  
    if dataset=='pc':
        image = image_data['pavia']   
        label = label_data['pavia_gt']  
    
    image = image.astype(np.float32 )     #归一化
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    shape = np.shape(image)    #填充数据，方便取块
    imag1=np.zeros((shape[0]+windowsize,shape[1]+windowsize,shape[2]), dtype=np.float32)
    half=windowsize//2
    imag1[half:half+shape[0] ,half:half+shape[1] ,:]=image  #内部填充

    # imag1[0:half,half:half+shape[1] ,:]=image[0:half,:,:] #左右填充
    # imag1[half+shape[0]+1:shape[0]+windowsize ,half:half+shape[1] ,:]=image[shape[0]-half:shape[0],:,:]  
    # imag1[half:half+shape[0],0:half,:]=image[:,0:half,:] #下上填充
    # imag1[half:half+shape[0] ,half+shape[1]+1:shape[1]+windowsize ,:]=image[:,shape[1]-half:shape[1],:] 
    # shape1 = np.shape(imag1)   
    # imag1[0:half,0:half,:]=image[0:half,0:half,:] #四个角填充
    # imag1[0:half,shape1[1]-half:shape1[1],:]=image[0:half,shape[1]-half:shape[1],:]
    # imag1[shape1[0]-half:shape1[0],0:half,:]=image[shape[0]-half:shape[0],0:half,:]
    # imag1[shape1[0]-half:shape1[0],shape1[1]-half:shape1[1],:]=image[shape[0]-half:shape[0],shape[1]-half:shape[1],:]
   
    # print(label.dtype)
    shape1 = np.shape(label)    
    label1=np.zeros((shape1[0]+windowsize,shape1[1]+windowsize),dtype=np.uint8)
    half=windowsize//2
    label1[half:half+shape1[0] ,half:half+shape1[1]]=label
    return imag1, label1


class DataSet(object):

    def __init__(self, images, labels, dtype=dtypes.float32, reshape=False):

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint16, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)

        self._num_examples = images.shape[0]

        if reshape:
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2], images.shape[3])

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def one_hot_transform(x, length):
    ont_hot_array = np.zeros([1, length])
    ont_hot_array[0, int(x)-1] = 1
    return ont_hot_array


def get_dist_per(label_map, train_num, val_num):
    category_num = label_map.max()
    train_per_num = train_num
    val_per_num =val_num
    train_dist = np.zeros(category_num)
    val_dist = np.zeros(category_num)
    cate_i_dist = np.zeros(category_num)

    for i in range(1, category_num):
        cate_i_num = sum(sum(label_map==i))
        if(cate_i_num==20):
            k=5
        else:
            k=0
        train_dist[i-1]=int(min(np.floor(cate_i_num//2), train_per_num))
        val_dist[i-1]=int(min(np.floor(cate_i_num//2-k), val_per_num))
        cate_i_dist[i-1]=cate_i_num

    train_dist[-1] = train_num
    val_dist[-1] = val_num
    cate_i_dist[-1] = sum(sum(label_map==category_num))
    # print(train_dist,val_dist)

    [m, n]=label_map.shape
    train_label_map = np.zeros((m,n))
    val_label_map = np.zeros((m,n))
    test_label_map = np.zeros((m,n))
    for i, [train_i_num, val_i_num, i_num] in enumerate(zip(train_dist, val_dist, cate_i_dist)):
        i_index = np.where(label_map==(i+1))
        shuffle_indices=np.random.permutation(int(i_num))
        train_indices=(i_index[0][shuffle_indices[:int(train_i_num)]], i_index[1][shuffle_indices[:int(train_i_num)]])
        val_indices=(i_index[0][shuffle_indices[int(train_i_num):int(train_i_num+val_i_num)]], i_index[1][shuffle_indices[int(train_i_num):int(train_i_num+val_i_num)]])
        test_indices=(i_index[0][shuffle_indices[int(train_i_num+val_i_num):]], i_index[1][shuffle_indices[int(train_i_num+val_i_num):]])

        train_label_map[train_indices]=i+1
        val_label_map[val_indices]=i+1
        test_label_map[test_indices]=i+1
    
    return train_label_map,val_label_map,test_label_map

def get_dist_all(label_map, train_num, val_num):
    label_num = sum(sum(label_map>0))
    train_ratio = train_num/label_num
    print('train_ratio',train_ratio)
    val_ratio = val_num/label_num
    print('val_ratio',val_ratio)
    category_num = label_map.max()
    train_dist = np.zeros(category_num)
    val_dist = np.zeros(category_num)
    cate_i_dist = np.zeros(category_num)

    for i in range(1, category_num):
        cate_i_num = sum(sum(label_map==i))
        train_dist[i-1]=max(2, np.around(train_ratio*cate_i_num))
        val_dist[i-1]=max(1, np.around(val_ratio*cate_i_num))
        cate_i_dist[i-1]=cate_i_num

    train_dist[-1] = train_num-train_dist.sum()
    val_dist[-1] = val_num-val_dist.sum()
    cate_i_dist[-1] = sum(sum(label_map==category_num))

    [m, n]=label_map.shape
    train_label_map = np.zeros((m,n))
    val_label_map = np.zeros((m,n))
    test_label_map = np.zeros((m,n))
    for i, [train_i_num, val_i_num, i_num] in enumerate(zip(train_dist, val_dist, cate_i_dist)):
        i_index = np.where(label_map==(i+1))
        shuffle_indices=np.random.permutation(int(i_num))
        train_indices=(i_index[0][shuffle_indices[:int(train_i_num)]], i_index[1][shuffle_indices[:int(train_i_num)]])
        val_indices=(i_index[0][shuffle_indices[int(train_i_num):int(train_i_num+val_i_num)]], i_index[1][shuffle_indices[int(train_i_num):int(train_i_num+val_i_num)]])
        test_indices=(i_index[0][shuffle_indices[int(train_i_num+val_i_num):]], i_index[1][shuffle_indices[int(train_i_num+val_i_num):]])

        train_label_map[train_indices]=i+1
        val_label_map[val_indices]=i+1
        test_label_map[test_indices]=i+1
    
    not_zero_raw1, not_zero_col1 = train_label_map.nonzero()    #非零位置的行列

    not_zero_raw2, not_zero_col2 = val_label_map.nonzero()    #非零位置的行列

    not_zero_raw, not_zero_col = test_label_map.nonzero()    #非零位置的行列

    t_samples=len(not_zero_col1)
    v_samples=len(not_zero_col2)
    # test_nsamples = number_samples - t_samples - v_samples
    print(t_samples)
    print(v_samples)
    return train_label_map,val_label_map,test_label_map

def readdata(image_file, label_file, datasetm, train_nsamples=200, validation_nsamples=100, windowsize=27,
             istraining=True, shuffle_number=None, batchnumber=10000, times=0):

    image, label = load_data(image_file, label_file, windowsize, datasetm)
    shape = np.shape(image)
    halfsize = windowsize // 2
    number_class = np.max(label)
    
    not_zero_raw, not_zero_col = label.nonzero()    #非零位置的行列
    number_samples = len(not_zero_raw)
    print('总共非零的样本数',number_samples)

    train_label_map,val_label_map,test_label_map=get_dist_per(label, train_nsamples, validation_nsamples)

    return  train_label_map,val_label_map,test_label_map
    


def main():
    '''生成训练、验证、测试样本集'''
    datasetm='houston'   #houston ksc pu pc  
    if datasetm=='houston':
        image_file = r'data/houston/houston.mat'   
        label_file = r'data/houston/houston_gt_sum.mat'   
    if datasetm=='ksc':
        image_file = r'data/ksc/KSC.mat'   
        label_file = r'data/ksc/KSC_gt.mat'   
    if datasetm=='pu':
        image_file = r'data/paviaU/PaviaU.mat'  
        label_file = r'data/paviaU/PaviaU_gt.mat'   
    if datasetm=='pc':
        image_file = r'data/paviaC/Pavia.mat'   
        label_file = r'data/paviaC/Pavia_gt.mat'   

    train_nsamples=20
    validation_nsamples=10
    windowsize=27
    train_label_map,val_label_map,test_label_map = readdata(image_file, label_file, datasetm, train_nsamples, validation_nsamples,
                                  windowsize)
    with h5py.File(os.path.join(args.data_root, datasetm + '_dist_per_train-{}_val-{}.h5'.format(train_nsamples, validation_nsamples)), 'w') as f:
       
        f['train_label_map'] = train_label_map,
        f['val_label_map'] = val_label_map,
        f['test_label_map'] = test_label_map,



if __name__ == '__main__':
    main()
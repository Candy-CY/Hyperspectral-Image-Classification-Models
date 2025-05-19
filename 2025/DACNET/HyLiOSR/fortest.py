# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:15:42 2023

@author: ryxax
"""

import os
import argparse
from keras.callbacks import EarlyStopping
from keras import losses
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import Adadelta
from keras.optimizers import Adadelta
from sklearn.metrics.pairwise import paired_distances as dist
from Hyper import imgDraw, listClassification, resnet99_avg_recon
import libmr
import numpy as np
import rscls
import glob
from scipy import io
from copy import deepcopy
import time
os.environ['CUDA_VISIBLE_DEVICES']='1'
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--numTrain', type=int, default=20)  # number of training samples per class,small than 40
parser.add_argument('--dataset', type=str, default='data/Trento/trento_im.npy')  # dataset name/home/amax/data/xibo/OpenSSLR/PaviaU/paviau_gt_wopen.txt
# parser.add_argument('--dataset', type=str, default='data/Houston/houston_im.npy')  # dataset name/home/amax/data/xibo/OpenSSLR/PaviaU/paviau_gt_wopen.txt

parser.add_argument('--gt', type=str, default='data/Trento/trento_raw_gt.npy')  # only known training samples included
# parser.add_argument('--gt', type=str, default='data/Houston/houston_raw_gt.npy')  # only known training samples included

parser.add_argument('--batch_size', type=int, default=16)  # only known training samples included
parser.add_argument('--output', type=str, default='output/')  # save path for output files
args = parser.parse_args()

# generate output dir
early_stopping = EarlyStopping(monitor='loss', patience=1000)
key = args.dataset.split('/')[-1].split('_')[0]
spath = args.output + key + '_' + str(args.numTrain) + '/'
os.makedirs(spath, exist_ok=True)

# load dataset
print('Preparing dataset...')
hsi = np.load(args.dataset).astype('float32')
gt = np.load(args.gt).astype('int')
listClassification(gt)
numClass = gt.max()
row, col, layers = hsi.shape
hsi = np.float32(hsi)

# dataset format
c1 = rscls.rscls(hsi, gt, cls=numClass)
c1.padding(9)
x1_train, y1_train = c1.train_sample(args.numTrain)  # load train samples
x1_train, y1_train = rscls.make_sample(x1_train, y1_train)  # augmentation
x1_lidar_train = x1_train[:, :, :, -1]
x1_lidar_train = x1_lidar_train[:, :, :, np.newaxis]
x1_train = x1_train[:, :, :, :-1]
y1_train = to_categorical(y1_train, numClass)  # to one-hot labels
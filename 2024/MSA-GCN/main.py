# -*- coding:utf-8 -*-
# Classification of Hyperspectral and LiDAR Data
import torch
from dataset import load_data, generater, normalization, setup_seed

from train import *
from test import *
from module import applyPCA
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ['Trento', 'MUUFL', 'Houston']

batch_sizes = [32, 64, 32]
windowSize = [15, 15, 15]
num_classes = [6, 11, 15]
in_channels_2 = [1, 2, 1]
applyPCA_Channel = 32

i=0


HSI_data, LiDAR_data, Train_data, Test_data, GT = load_data(dataset[i])
# 归一化
print(HSI_data.shape)
print(LiDAR_data.shape)

HSI_data, _ = applyPCA(HSI_data, numComponents=applyPCA_Channel)

LiDAR_data = normalization(LiDAR_data, type=1)
HSI_data = normalization(HSI_data, type=1)


TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, train_iter, test_iter, gt_iter, y_gt = generater(HSI_data,
                                                                     LiDAR_data,
                                                                     Train_data,
                                                                     Test_data,
                                                                     GT,
                                                                     batch_size=batch_sizes[i],
                                                                     windowSize=windowSize[i]
                                                                     )


# model1 = train_best_model(dataset=dataset[i],
#      train_iter=train_iter,
#      device=device,
#      epoches=150,
#      ITER=1,
#      TRAIN_SIZE=TRAIN_SIZE,
#      TEST_SIZE=TEST_SIZE,
#      TOTAL_SIZE=TOTAL_SIZE,
#      test_iter=test_iter,
#      num_classes=num_classes[i],
#      in_channels_2=in_channels_2[i],
#      windowSize=windowSize[i],
#      applyPCA_Channel=applyPCA_Channel)

model1 = Feature_HSI_Lidar(applyPCA_Channel, in_channels_2[i], num_classes[i], windowSize[i]).to(device)


classification, oa, aa, kappa, each_acc = test(test_iter=test_iter,
                                               device=device,
                                               dataset=dataset[i],
                                               net1=model1)
print(classification, oa, aa, kappa, each_acc)




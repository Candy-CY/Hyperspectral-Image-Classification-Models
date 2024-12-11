# -*- coding:utf-8 -*-

from module import split_train_test_set, pixel_select, GetImageCubes, GetImageCubes_all

import random

import numpy as np
import scipy.io as sio
import os

import torch
import torch.utils.data as Data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(dataset):
    #　data_path = r'/home/ubuntu/dataset_RS/Multisource/data'
    # data_path = r'F:\science\data'
    data_path = r'F:\science\data_server'

    if dataset == 'Houston':  # HSI.shape (349, 1905, 144), LiDAR.shape (349, 1905) gt.shape (349, 1905)=
        HSI_data = sio.loadmat(os.path.join(data_path, 'Houston2013_Data/Houston2013_HSI.mat'))['HSI']
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'Houston2013_Data/Houston2013_DSM.mat'))['DSM']
        LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)
        Train_data = sio.loadmat(os.path.join(data_path, 'Houston2013_Data/Houston2013_TR.mat'))['TR_map']
        Test_data = sio.loadmat(os.path.join(data_path, 'Houston2013_Data/Houston2013_TE.mat'))['TE_map']
        GT = sio.loadmat(os.path.join(data_path, 'Houston2013_Data/gt.mat'))['gt']

    if dataset == 'Berlin':
        HSI_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/data_HS_LR.mat'))['data_HS_LR']
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/data_SAR_HR.mat'))['data_SAR_HR']
        Train_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/TrainImage.mat'))['TrainImage']
        Test_data = sio.loadmat(os.path.join(data_path, 'HS-SAR Berlin/TestImage.mat'))['TestImage']

    if dataset == 'Augsburg':
        HSI_data = sio.loadmat(os.path.join(data_path, 'HS-SAR-DSM Augsburg/data_HS_LR.mat'))['data_HS_LR']
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'HS-SAR-DSM Augsburg/data_DSM.mat'))['data_DSM']
        Train_data = sio.loadmat(os.path.join(data_path, 'HS-SAR-DSM Augsburg/TrainImage.mat'))['TrainImage']
        Test_data = sio.loadmat(os.path.join(data_path, 'HS-SAR-DSM Augsburg/TestImage.mat'))['TestImage']
        GT=sio.loadmat(os.path.join(data_path, 'HS-SAR-DSM Augsburg/gt.mat'))['gt']

    if dataset == 'Trento':
        HSI_data = sio.loadmat(os.path.join(data_path, 'Trento/HSI.mat'))['HSI']
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'Trento/LiDAR.mat'))['LiDAR']
        LiDAR_data = np.expand_dims(LiDAR_data, axis=-1)
        Train_data = sio.loadmat(os.path.join(data_path, 'Trento/TRLabel.mat'))['TRLabel']
        Test_data = sio.loadmat(os.path.join(data_path, 'Trento/TSLabel.mat'))['TSLabel']
        GT = sio.loadmat(os.path.join(data_path, 'Trento/gt.mat'))['gt']

    if dataset == 'MUUFL':
        HSI_data = sio.loadmat(os.path.join(data_path, 'MUUFL/HSI.mat'))['HSI']
        LiDAR_data = sio.loadmat(os.path.join(data_path, 'MUUFL/LiDAR.mat'))['LiDAR']
        Train_data = sio.loadmat(os.path.join(data_path, 'MUUFL/mask_train_150.mat'))['mask_train']
        Test_data = sio.loadmat(os.path.join(data_path, 'MUUFL/mask_test_150.mat'))['mask_test']
        GT = sio.loadmat(os.path.join(data_path, 'MUUFL/gt.mat'))['gt']
        GT[GT == -1] = 0
    return HSI_data, LiDAR_data, Train_data, Test_data, GT


def generater(X_hsi, X_lidar, train_pixels, test_pixels, gt, batch_size, windowSize):
    #　train_pixels, _ = pixel_select(train_pixels, train_ratio=0.3)

    x_train_hsi, x_train_hsi_row, x_train_hsi_col, y_train_hsi, position_train_hsi = GetImageCubes(input_data=X_hsi,
                                                                                                   pixels_select=train_pixels,
                                                                                                   windowSize=windowSize)
    x_test_hsi, x_test_hsi_row, x_test_hsi_col, y_test_hsi, position_test_hsi = GetImageCubes(input_data=X_hsi,
                                                                                              pixels_select=test_pixels,
                                                                                              windowSize=windowSize)
    x_gt_hsi, x_gt_hsi_row, x_gt_hsi_col, y_gt_hsi, position_gt_hsi = GetImageCubes(input_data=X_hsi, pixels_select=gt,
                                                                                    windowSize=windowSize)
    # x_all_hsi = GetImageCubes_all(input_data=X_hsi, pixels_select=gt, windowSize=windowSize)

    x_train_lidar, x_train_lidar_row, x_train_lidar_col, y_train_lidar, position_train_lidar = GetImageCubes(
        input_data=X_lidar, pixels_select=train_pixels, windowSize=windowSize)
    x_test_lidar, x_test_lidar_row, x_test_lidar_col, y_test_lidar, position_test_lidar = GetImageCubes(
        input_data=X_lidar, pixels_select=test_pixels, windowSize=windowSize)
    x_gt_lidar, x_gt_lidar_row, x_gt_lidar_col, y_gt_lidar, position_gt_lidar = GetImageCubes(input_data=X_lidar,
                                                                                              pixels_select=gt,
                                                                                              windowSize=windowSize)
    # x_all_lidar = GetImageCubes_all(input_data=X_lidar, pixels_select=gt, windowSize=windowSize)

    TRAIN_SIZE = x_train_hsi.shape[0]
    TEST_SIZE = x_test_hsi.shape[0]
    TOTAL_SIZE = y_train_hsi.shape[0] + y_test_hsi.shape[0]

    print('X_train:{}\nX_test:{}\nX_all:{}'.format(x_train_hsi.shape[0], x_test_hsi.shape[0],
                                                   x_train_hsi.shape[0] + x_test_hsi.shape[0]))

    hsi_train_tensor = torch.from_numpy(x_train_hsi).type(torch.FloatTensor)
    hsi_train_row_tensor = torch.from_numpy(x_train_hsi_row).type(torch.FloatTensor)
    hsi_train_col_tensor = torch.from_numpy(x_train_hsi_col).type(torch.FloatTensor)
    position_train_hsi_tensor = torch.from_numpy(position_train_hsi).type(torch.FloatTensor)
    hsi_test_tensor = torch.from_numpy(x_test_hsi).type(torch.FloatTensor)
    hsi_test_row_tensor = torch.from_numpy(x_test_hsi_row).type(torch.FloatTensor)
    hsi_test_col_tensor = torch.from_numpy(x_test_hsi_col).type(torch.FloatTensor)
    position_test_hsi_tensor = torch.from_numpy(position_test_hsi).type(torch.FloatTensor)

    hsi_gt_tensor = torch.from_numpy(x_gt_hsi).type(torch.FloatTensor)
    hsi_gt_row_tensor = torch.from_numpy(x_gt_hsi_row).type(torch.FloatTensor)
    hsi_gt_col_tensor = torch.from_numpy(x_gt_hsi_col).type(torch.FloatTensor)
    # hsi_all_tensor = torch.from_numpy(x_all_hsi).type(torch.FloatTensor)

    lidar_train_tensor = torch.from_numpy(x_train_lidar).type(torch.FloatTensor)
    lidar_train_row_tensor = torch.from_numpy(x_train_lidar_row).type(torch.FloatTensor)
    lidar_train_col_tensor = torch.from_numpy(x_train_lidar_col).type(torch.FloatTensor)
    position_train_lidar_tensor = torch.from_numpy(position_train_lidar).type(torch.FloatTensor)
    lidar_test_tensor = torch.from_numpy(x_test_lidar).type(torch.FloatTensor)
    lidar_test_row_tensor = torch.from_numpy(x_test_lidar_row).type(torch.FloatTensor)
    lidar_test_col_tensor = torch.from_numpy(x_test_lidar_col).type(torch.FloatTensor)
    position_test_lidar_tensor = torch.from_numpy(position_test_lidar).type(torch.FloatTensor)

    lidar_gt_tensor = torch.from_numpy(x_gt_lidar).type(torch.FloatTensor)
    lidar_gt_row_tensor = torch.from_numpy(x_gt_lidar_row).type(torch.FloatTensor)
    lidar_gt_col_tensor = torch.from_numpy(x_gt_lidar_col).type(torch.FloatTensor)
    # lidar_all_tensor = torch.from_numpy(x_all_lidar).type(torch.FloatTensor)

    y_train = torch.from_numpy(y_train_hsi).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test_hsi).type(torch.FloatTensor)
    y_gt = torch.from_numpy(y_gt_hsi).type(torch.FloatTensor)

    torch_train = Data.TensorDataset(hsi_train_tensor, hsi_train_row_tensor, hsi_train_col_tensor, lidar_train_tensor,
                                     lidar_train_row_tensor, lidar_train_col_tensor, y_train, position_train_hsi_tensor)
    torch_test = Data.TensorDataset(hsi_test_tensor, hsi_test_row_tensor, hsi_test_col_tensor, lidar_test_tensor,
                                    lidar_test_row_tensor, lidar_test_col_tensor, y_test, position_test_hsi_tensor)
    torch_gt = Data.TensorDataset(hsi_gt_tensor, hsi_gt_row_tensor, hsi_gt_col_tensor, lidar_gt_tensor,
                                     lidar_gt_row_tensor, lidar_gt_col_tensor, y_gt)
    # torch_all = Data.TensorDataset(hsi_all_tensor, lidar_all_tensor)

    train_iter = Data.DataLoader(
        dataset=torch_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        # drop_last=True
    )

    test_iter = Data.DataLoader(
        dataset=torch_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    gt_iter = Data.DataLoader(
        dataset=torch_gt,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    '''
    all_iter = Data.DataLoader(
        dataset=torch_all,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    '''
    # return TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, train_iter, test_iter, all_iter, gt_iter
    return TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, train_iter, test_iter, gt_iter,y_test_hsi


# HSI_data, LiDAR_data, gt = load_data('Houston') #HSI.shape (349, 1905, 144), LiDAR.shape (349, 1905) gt.shape (349, 1905)
#
# generater(HSI_data, LiDAR_data, gt, 0.1, 15)

def normalization(X, type=1):
    x = np.zeros(shape=X.shape, dtype='float32')
    if type == 1:
        for i in range(X.shape[2]):
            temp = X[:, :, i]
            mean = np.mean(temp)
            std = np.std(temp)
            x[:, :, i] = ((temp - mean) / std)
    if type == 2:
        for i in range(X.shape[2]):
            min = np.min(X[:, :, i])
            max = np.max(X[:, :, i])
            scale = max - min
            if scale == 0:
                scale = 1e-5
            x[:, :, i] = (X[:, :, i] - min) / scale
    return x

# HSI_data, LiDAR_data, gt = load_data('Houston') #HSI.shape (349, 1905, 144), LiDAR.shape (349, 1905) gt.shape (349, 1905)
#
# generater(HSI_data, LiDAR_data, gt, 0.1, 15)

# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2022-Nov
# @Address  : Time Lab @ SDU
# @FileName : data_load_operate.py
# @Project  : CVSSN (HSIC), IEEE TCSVT

import os
import torch
import numpy as np
import scipy.io as sio
import torch.utils.data as Data
from sklearn import preprocessing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(data_set_name, data_path):
    if data_set_name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'IP', 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'IP', 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif data_set_name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC', 'KSC.mat'))['KSC']
        # data = sio.loadmat(os.path.join(data_path, 'KSC', 'KSC_corrected.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC', 'KSC_gt.mat'))['KSC_gt']
    elif data_set_name == 'UP':
        data = sio.loadmat(os.path.join(data_path, 'PU', 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PU', 'PaviaU_gt.mat'))['paviaU_gt']

    return data, labels


def load_HU_data(data_set_name, data_path):
    if data_set_name == 'HU_tif':
        data = sio.loadmat(os.path.join(data_path, 'HU13_tif', "Houston13_data.mat"))['Houston13_data']
        labels_train = sio.loadmat(os.path.join(data_path, 'HU13_tif', "Houston13_gt_train.mat"))['Houston13_gt_train']
        labels_test = sio.loadmat(os.path.join(data_path, 'HU13_tif', "Houston13_gt_test.mat"))['Houston13_gt_test']

    return data, labels_train, labels_test


def standardization(data):
    height, width, bands = data.shape
    data = np.reshape(data, [height * width, bands])
    data = preprocessing.StandardScaler().fit_transform(data)

    data = np.reshape(data, [height, width, bands])
    return data


def data_pad_zero(data, patch_length):
    data_padded = np.lib.pad(data, ((patch_length, patch_length), (patch_length, patch_length), (0, 0)), 'constant',
                             constant_values=0)
    return data_padded


def sampling(ratio_list, num_list, gt_reshape, class_count, Flag):
    all_label_index_dict, train_label_index_dict, val_label_index_dict, test_label_index_dict = {}, {}, {}, {}
    all_label_index_list, train_label_index_list, val_label_index_list, test_label_index_list = [], [], [], []

    for cls in range(class_count):
        cls_index = np.where(gt_reshape == cls + 1)[0]
        all_label_index_dict[cls] = list(cls_index)

        np.random.shuffle(cls_index)

        if Flag == 0:  # Fixed proportion for each category
            train_index_flag = max(int(ratio_list[0] * len(cls_index)), 3)  # at least 3 samples per class]
            val_index_flag = max(int(ratio_list[1] * len(cls_index)), 1)
        # Split by num per class
        elif Flag == 1:  # Fixed quantity per category
            if len(cls_index) > num_list[0]:
                train_index_flag = num_list[0]
            else:
                train_index_flag = 15
            val_index_flag = num_list[1]

        train_label_index_dict[cls] = list(cls_index[:train_index_flag])
        test_label_index_dict[cls] = list(cls_index[train_index_flag:][val_index_flag:])
        val_label_index_dict[cls] = list(cls_index[train_index_flag:][:val_index_flag])

        train_label_index_list += train_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]
        val_label_index_list += val_label_index_dict[cls]
        all_label_index_list += all_label_index_dict[cls]

    return train_label_index_list, val_label_index_list, test_label_index_list, all_label_index_list


# Create index table mapping from 1D to 2D, from unpadded index to padded index
def index_assignment(index, row, col, pad_length):
    new_assign = {}  # dictionary.
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def select_patch(data_padded, pos_x, pos_y, patch_length):
    selected_patch = data_padded[pos_x - patch_length:pos_x + patch_length + 1,
                     pos_y - patch_length:pos_y + patch_length + 1]
    return selected_patch


def select_vector(data_padded, pos_x, pos_y):
    select_vector = data_padded[pos_x, pos_y]
    return select_vector


def HSI_create_pathes(data_padded, hsi_h, hsi_w, data_indexes, patch_length, flag):
    h_p, w_p, c = data_padded.shape

    data_size = len(data_indexes)
    patch_size = patch_length * 2 + 1

    data_assign = index_assignment(data_indexes, hsi_h, hsi_w, patch_length)
    if flag == 1:
        # for spatial net data, HSI patch
        unit_data = np.zeros((data_size, patch_size, patch_size, c))
        unit_data_torch = torch.from_numpy(unit_data).type(torch.FloatTensor).to(device)
        for i in range(len(data_assign)):
            unit_data_torch[i] = select_patch(data_padded, data_assign[i][0], data_assign[i][1], patch_length)

    if flag == 2:
        # for spectral net data, HSI vector
        unit_data = np.zeros((data_size, c))
        unit_data_torch = torch.from_numpy(unit_data).type(torch.FloatTensor).to(device)
        for i in range(len(data_assign)):
            unit_data_torch[i] = select_vector(data_padded, data_assign[i][0], data_assign[i][1])

    return unit_data_torch


# generating HSI patches using GPU directly.
def generate_iter_1(data_padded, hsi_h, hsi_w, label_reshape, index, patch_length, batch_size, model_type_flag,
                    model_3D_spa_flag):
    # flag for single spatial net or single spectral net or spectral-spatial net
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(device)

    # for data label
    train_labels = label_reshape[index[0]] - 1
    val_labels = label_reshape[index[1]] - 1
    test_labels = label_reshape[index[2]] - 1

    y_tensor_train = torch.from_numpy(train_labels).type(torch.FloatTensor)
    y_tensor_val = torch.from_numpy(val_labels).type(torch.FloatTensor)
    y_tensor_test = torch.from_numpy(test_labels).type(torch.FloatTensor)

    # for data
    if model_type_flag == 1:  # data for single spatial net
        spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
        spa_val_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)
        spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[2], patch_length, 1)

        if model_3D_spa_flag == 1:  # spatial 3D patch
            spa_train_samples = spa_train_samples.unsqueeze(1)
            spa_test_samples = spa_test_samples.unsqueeze(1)
            spa_val_samples = spa_val_samples.unsqueeze(1)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, y_tensor_train)
        torch_dataset_val = Data.TensorDataset(spa_val_samples, y_tensor_val)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, y_tensor_test)

    elif model_type_flag == 2:  # data for single spectral net
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_val_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[2], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spe_train_samples, y_tensor_train)
        torch_dataset_val = Data.TensorDataset(spe_val_samples, y_tensor_val)
        torch_dataset_test = Data.TensorDataset(spe_test_samples, y_tensor_test)

    elif model_type_flag == 3:  # data for spectral-spatial net
        # spatail data
        spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
        spa_val_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)
        spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[2], patch_length, 1)

        # spectral data
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_val_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[2], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, spe_train_samples, y_tensor_train)
        torch_dataset_val = Data.TensorDataset(spa_val_samples, spe_val_samples, y_tensor_val)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, spe_test_samples, y_tensor_test)

    train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_iter = Data.DataLoader(dataset=torch_dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter, val_iter


# 1) generating HSI patches for the visualization of all the labeled samples of the data set
# 2) generating HSI patches for the visualization of total the samples of the data set
# in addition, 1) and 2) both use GPU directly
def generate_iter_2(data_padded, hsi_h, hsi_w, label_reshape, index, patch_length, batch_size, model_type_flag,
                    model_3D_spa_flag):
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(device)

    if len(index) < label_reshape.shape[0]:
        total_labels = label_reshape[index] - 1
    else:
        total_labels = np.zeros(label_reshape.shape)

    y_tensor_total = torch.from_numpy(total_labels).type(torch.FloatTensor)

    if model_type_flag == 1:
        total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 1)
        if model_3D_spa_flag == 1:  # spatial 3D patch
            total_samples = total_samples.unsqueeze(1)
        torch_dataset_total = Data.TensorDataset(total_samples, y_tensor_total)

    elif model_type_flag == 2:
        total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 2)
        torch_dataset_total = Data.TensorDataset(total_samples, y_tensor_total)
    elif model_type_flag == 3:
        spa_total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 1)
        spe_total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 2)
        torch_dataset_total = Data.TensorDataset(spa_total_samples, spe_total_samples, y_tensor_total)

    total_iter = Data.DataLoader(dataset=torch_dataset_total, batch_size=batch_size, shuffle=False, num_workers=0)

    return total_iter


def generate_data_set(data_reshape, label, index):
    train_data_index, test_data_index, all_data_index = index

    x_train_set = data_reshape[train_data_index]
    y_train_set = label[train_data_index] - 1

    x_test_set = data_reshape[test_data_index]
    y_test_set = label[test_data_index] - 1

    x_all_set = data_reshape[all_data_index]
    y_all_set = label[all_data_index] - 1

    return x_train_set, y_train_set, x_test_set, y_test_set, x_all_set, y_all_set


def generate_data_set_hu(data_reshape, label_train, label_test, index):
    train_data_index, test_data_index, all_data_index = index

    x_train_set = data_reshape[train_data_index]
    y_train_set = label_train[train_data_index] - 1

    x_test_set = data_reshape[test_data_index]
    y_test_set = label_test[test_data_index] - 1

    return x_train_set, y_train_set, x_test_set, y_test_set

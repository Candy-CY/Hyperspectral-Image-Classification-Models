# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2022-Nov
# @Address  : Time Lab @ SDU
# @FileName : data_load_operate_disjoint.py
# @Project  : CVSSN (HSIC), IEEE TCSVT

import torch
import numpy as np
import torch.utils.data as Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sampling_UH_w_val(gt_train_re, gt_test_re, class_count):
    all_label_index_dict, train_label_index_dict, val_label_index_dict, test_label_index_dict = {}, {}, {}, {}
    all_label_index_list, train_label_index_list, val_label_index_list, test_label_index_list = [], [], [], []
    val_len = 3
    for cls in range(class_count):
        cls_index_train = np.where(gt_train_re == cls + 1)[0]
        cls_index_test = np.where(gt_test_re == cls + 1)[0]

        np.random.shuffle(cls_index_test)

        train_label_index_dict[cls] = list(cls_index_train)
        val_label_index_dict[cls] = list(cls_index_test[:val_len])
        test_label_index_dict[cls] = list(cls_index_test[val_len:])

        train_label_index_list += train_label_index_dict[cls]
        val_label_index_list += val_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]

        all_label_index_list += (train_label_index_dict[cls] + test_label_index_dict[cls])

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


def HSI_create_pathes(data_padded, data, data_indexes, patch_length, flag):
    h_p, w_p, c = data_padded.shape

    data_size = len(data_indexes)
    patch_size = patch_length * 2 + 1

    data_assign = index_assignment(data_indexes, data.shape[0], data.shape[1], patch_length)
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


def generate_iter_hu_w_val(data_padded, data, gt_train_re, gt_test_re, index, patch_length, batch_size, model_type_flag,
                           model_3D_spa_flag):
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(device)
    data_torch = torch.from_numpy(data).type(torch.FloatTensor).to(device)

    # for data label
    train_labels = gt_train_re[index[0]] - 1
    val_labels = gt_test_re[index[1]] - 1
    test_labels = gt_test_re[index[2]] - 1

    y_tensor_train = torch.from_numpy(train_labels).type(torch.FloatTensor)
    y_tensor_val = torch.from_numpy(val_labels).type(torch.FloatTensor)
    y_tensor_test = torch.from_numpy(test_labels).type(torch.FloatTensor)

    # for data
    if model_type_flag == 1:  # data for single spatial net
        spa_train_samples = HSI_create_pathes(data_padded_torch, data_torch, index[0], patch_length, 1)
        spa_val_samples = HSI_create_pathes(data_padded_torch, data_torch, index[1], patch_length, 1)
        spa_test_samples = HSI_create_pathes(data_padded_torch, data_torch, index[2], patch_length, 1)

        if model_3D_spa_flag == 1:  # spatial 3D patch
            spa_train_samples = spa_train_samples.unsqueeze(1)
            spa_test_samples = spa_test_samples.unsqueeze(1)
            spa_val_samples = spa_val_samples.unsqueeze(1)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, y_tensor_train)
        torch_dataset_val = Data.TensorDataset(spa_val_samples, y_tensor_val)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, y_tensor_test)

    elif model_type_flag == 2:  # data for single spectral net
        spe_train_samples = HSI_create_pathes(data_padded_torch, data_torch, index[0], patch_length, 2)
        spe_val_samples = HSI_create_pathes(data_padded_torch, data_torch, index[1], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, data_torch, index[2], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spe_train_samples, y_tensor_train)
        torch_dataset_val = Data.TensorDataset(spe_val_samples, y_tensor_val)
        torch_dataset_test = Data.TensorDataset(spe_test_samples, y_tensor_test)

    elif model_type_flag == 3:  # data for spectral-spatial net
        # spatail data
        spa_train_samples = HSI_create_pathes(data_padded_torch, data_torch, index[0], patch_length, 1)
        spa_val_samples = HSI_create_pathes(data_padded_torch, data_torch, index[1], patch_length, 1)
        spa_test_samples = HSI_create_pathes(data_padded_torch, data_torch, index[2], patch_length, 1)

        # spectral data
        spe_train_samples = HSI_create_pathes(data_padded_torch, data_torch, index[0], patch_length, 2)
        spe_val_samples = HSI_create_pathes(data_padded_torch, data_torch, index[1], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, data_torch, index[2], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, spe_train_samples, y_tensor_train)
        torch_dataset_val = Data.TensorDataset(spa_val_samples, spe_val_samples, y_tensor_val)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, spe_test_samples, y_tensor_test)

    train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_iter = Data.DataLoader(dataset=torch_dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, val_iter, test_iter

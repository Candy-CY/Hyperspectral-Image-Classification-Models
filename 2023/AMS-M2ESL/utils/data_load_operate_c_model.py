# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : data_load_operate_c_model.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

import os
import math
import torch
import numpy as np
import spectral as spy
import scipy.io as sio
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(data_set_name, data_path):
    if data_set_name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'IP', 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'IP', 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif data_set_name == 'UP':
        data = sio.loadmat(os.path.join(data_path, 'UP', 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'UP', 'PaviaU_gt.mat'))['paviaU_gt']
    return data, labels


def load_HU_data(data_path):
    data = sio.loadmat(os.path.join(data_path, 'HU13_tif', "Houston13_data.mat"))['Houston13_data']
    labels_train = sio.loadmat(os.path.join(data_path, 'HU13_tif', "Houston13_gt_train.mat"))['Houston13_gt_train']
    labels_test = sio.loadmat(os.path.join(data_path, 'HU13_tif', "Houston13_gt_test.mat"))['Houston13_gt_test']

    return data, labels_train, labels_test


def standardization(data):
    height, width, bands = data.shape
    data = np.reshape(data, [height * width, bands])
    # data=preprocessing.scale(data) #
    # data = preprocessing.MinMaxScaler().fit_transform(data)
    data = preprocessing.StandardScaler().fit_transform(data)  #

    data = np.reshape(data, [height, width, bands])
    return data


def sampling(ratio_list, num_list, gt_reshape, class_count, Flag):
    all_label_index_dict, train_label_index_dict, test_label_index_dict = {}, {}, {}
    all_label_index_list, train_label_index_list, test_label_index_list = [], [], [],

    for cls in range(class_count):  # [0-15]
        cls_index = np.where(gt_reshape == cls + 1)[0]
        all_label_index_dict[cls] = list(cls_index)

        np.random.shuffle(cls_index)

        if Flag == 0:  # Fixed proportion for each category
            train_index_flag = max(int(ratio_list[0] * len(cls_index)), 3)  # at least 3 samples per class]
        # Split by num per class
        elif Flag == 1:  # Fixed quantity per category
            if len(cls_index) > num_list[0]:
                train_index_flag = num_list[0]
            else:
                train_index_flag = 15

        train_label_index_dict[cls] = list(cls_index[:train_index_flag])
        test_label_index_dict[cls] = list(cls_index[train_index_flag:])

        train_label_index_list += train_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]
        all_label_index_list += all_label_index_dict[cls]

    return train_label_index_list, test_label_index_list, all_label_index_list


def sampling_disjoint(gt_train_re, gt_test_re, class_count):
    all_label_index_dict, train_label_index_dict, test_label_index_dict = {}, {}, {}
    all_label_index_list, train_label_index_list, test_label_index_list = [], [], []

    for cls in range(class_count):
        cls_index_train = np.where(gt_train_re == cls + 1)[0]
        cls_index_test = np.where(gt_test_re == cls + 1)[0]

        train_label_index_dict[cls] = list(cls_index_train)
        test_label_index_dict[cls] = list(cls_index_test)

        train_label_index_list += train_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]
        all_label_index_list += (train_label_index_dict[cls] + test_label_index_dict[cls])

    return train_label_index_list, test_label_index_list, all_label_index_list


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def HSI_MNF(X, MNF_ratio):
    denoised_bands = math.ceil(MNF_ratio * X.shape[-1])
    mnfr = spy.mnf(spy.calc_stats(X), spy.noise_from_diffs(X))
    denoised_data = mnfr.reduce(X, num=denoised_bands)

    return denoised_data


def data_pad_zero(data, patch_length):
    data_padded = np.lib.pad(data, ((patch_length, patch_length), (patch_length, patch_length), (0, 0)), 'constant',
                             constant_values=0)
    return data_padded


def img_show(x):
    spy.imshow(x)
    plt.show()


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


def HSI_create_pathes_spatial_multiscale(data, data_indexes, scales):
    h, w, c = data.shape
    data_size = len(data_indexes)


    CR_data = np.zeros((data_size, scales, c, c))
    CR_data_torch = torch.from_numpy(CR_data).type(torch.FloatTensor).to('cuda')

    for j in range(scales):
        patch_length = j + 1
        patch_size = 2 * patch_length + 1

        data_padded = data_pad_zero(data, patch_length)
        data_assign = index_assignment(data_indexes, h, w, patch_length)

        unit_data = np.zeros((data_size, patch_size, patch_size, c))

        for i in range(data_size):
            unit_data[i] = select_patch(data_padded, data_assign[i][0], data_assign[i][1], patch_length)

        CR_j = Covar_cor_mat(unit_data)
        CR_j = torch.unsqueeze(CR_j, dim=1)
        CR_data_torch[:, j:, ] = CR_j

    return CR_data_torch


def Covar_cor_mat(x):
    x_t = torch.from_numpy(x).type(torch.FloatTensor).to('cuda')
    batch_size, h, w, c = x_t.size()

    x_t = x_t.view(batch_size, h * w, c)
    mean_pixel = torch.mean(x_t, dim=1, keepdims=True)
    mean_pixel_expand = mean_pixel.expand(x_t.shape)

    x_cr = x_t - mean_pixel_expand
    CR = torch.bmm(x_cr.permute(0, 2, 1), x_cr)
    CR = torch.div(CR, h * w - 1)
    del mean_pixel, mean_pixel_expand

    return CR


# generating HSI patches using GPU directly.
def generate_iter(data_padded, data, hsi_h, hsi_w, label_reshape, index, patch_length, batch_size,
                  model_type_flag,
                  model_3D_spa_flag, last_batch_flag, model_flag, dr_c, scales):
    # flag for single spatial net or single spectral net or spectral-spatial net
    if model_flag == 3:
        spa_dr_data = applyPCA(data_padded, dr_c)
        spa_dr_data_torch = torch.from_numpy(spa_dr_data).type(torch.FloatTensor).to(device)
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(device)

    # for data label
    train_labels = label_reshape[index[0]] - 1
    test_labels = label_reshape[index[1]] - 1

    y_tensor_train = torch.from_numpy(train_labels).type(torch.FloatTensor)
    y_tensor_test = torch.from_numpy(test_labels).type(torch.FloatTensor)

    # for data
    if model_type_flag == 1:  # data for single spatial net
        if model_flag == 1:
            spa_train_samples = HSI_create_pathes_spatial_multiscale(data, index[0], scales)
            spa_test_samples = HSI_create_pathes_spatial_multiscale(data, index[1], scales)
        else:
            spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
            spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)

        if model_3D_spa_flag == 1:  # spatial 3D patch
            spa_train_samples = spa_train_samples.unsqueeze(1)
            spa_test_samples = spa_test_samples.unsqueeze(1)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, y_tensor_test)

    elif model_type_flag == 2:  # data for single spectral net
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spe_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spe_test_samples, y_tensor_test)

    elif model_type_flag == 3:  # data for spectral-spatial net
        # spatail data
        if model_flag == 3:
            spa_train_samples = HSI_create_pathes(spa_dr_data_torch, hsi_h, hsi_w, index[0], patch_length, 1)
            spa_test_samples = HSI_create_pathes(spa_dr_data_torch, hsi_h, hsi_w, index[1], patch_length, 1)
        else:
            spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
            spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)

        # spectral data
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, spe_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, spe_test_samples, y_tensor_test)
    if last_batch_flag == 0:
        train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
        test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    elif last_batch_flag == 1:
        train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                     drop_last=True)
        test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0,
                                    drop_last=True)

    return train_iter, test_iter


def generate_iter_disjoint(data_padded, data, hsi_h, hsi_w, gt_train_re, gt_test_re, index, patch_length,
                           batch_size,
                           model_type_flag, model_3D_spa_flag, last_batch_flag, model_flag, dr_c, scales):
    if model_flag == 3:
        spa_dr_data = applyPCA(data_padded, dr_c)
        spa_dr_data_torch = torch.from_numpy(spa_dr_data).type(torch.FloatTensor).to("cuda")
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to("cuda")

    train_labels = gt_train_re[index[0]] - 1
    test_labels = gt_test_re[index[1]] - 1

    y_tensor_train = torch.from_numpy(train_labels).type(torch.FloatTensor)
    y_tensor_test = torch.from_numpy(test_labels).type(torch.FloatTensor)

    # for data
    if model_type_flag == 1:  # data for single spatial net
        if model_flag == 1:
            spa_train_samples = HSI_create_pathes_spatial_multiscale(data, index[0], scales)
            spa_test_samples = HSI_create_pathes_spatial_multiscale(data, index[1], scales)
        else:
            spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
            spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)

        if model_3D_spa_flag == 1:  # spatial 3D patch
            spa_train_samples = spa_train_samples.unsqueeze(1)
            spa_test_samples = spa_test_samples.unsqueeze(1)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, y_tensor_test)

    elif model_type_flag == 2:  # data for single spectral net
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spe_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spe_test_samples, y_tensor_test)

    elif model_type_flag == 3:  # data for spectral-spatial net
        # spatail data
        if model_flag == 3:
            spa_train_samples = HSI_create_pathes(spa_dr_data_torch, hsi_h, hsi_w, index[0], patch_length, 1)
            spa_test_samples = HSI_create_pathes(spa_dr_data_torch, hsi_h, hsi_w, index[1], patch_length, 1)
        else:
            spa_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 1)
            spa_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 1)

        # spectral data
        spe_train_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[0], patch_length, 2)
        spe_test_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index[1], patch_length, 2)

        torch_dataset_train = Data.TensorDataset(spa_train_samples, spe_train_samples, y_tensor_train)
        torch_dataset_test = Data.TensorDataset(spa_test_samples, spe_test_samples, y_tensor_test)

    if last_batch_flag == 0:
        train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
        test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    elif last_batch_flag == 1:
        train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                     drop_last=True)
        test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0,
                                    drop_last=True)

    return train_iter, test_iter


# 1) generating HSI patches for the visualization of all the labeled samples of the data set
# 2) generating HSI patches for the visualization of total the samples of the data set
# in addition, 1) and 2) both use GPU directly
def generate_iter_total(data_padded, data, hsi_h, hsi_w, label_reshape, index, patch_length, batch_size,
                        model_type_flag,
                        model_3D_spa_flag, last_batch_flag, model_flag, dr_c, scales):
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to("cuda")

    # flag for single spatial net or single spectral net or spectral-spatial net
    if model_flag == 3:
        spa_dr_data = applyPCA(data_padded, dr_c)
        spa_dr_data_torch = torch.from_numpy(spa_dr_data).type(torch.FloatTensor).to("cuda")

    if len(index) < label_reshape.shape[0]:
        total_labels = label_reshape[index] - 1
    else:
        total_labels = np.zeros(label_reshape.shape)

    y_tensor_total = torch.from_numpy(total_labels).type(torch.FloatTensor)

    if model_type_flag == 1:
        if model_flag == 1:
            total_samples = HSI_create_pathes_spatial_multiscale(data, index, scales)
        else:
            total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 1)
        if model_3D_spa_flag == 1:  # spatial 3D patch
            total_samples = total_samples.unsqueeze(1)
        torch_dataset_total = Data.TensorDataset(total_samples, y_tensor_total)
    elif model_type_flag == 2:
        total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 2)
        torch_dataset_total = Data.TensorDataset(total_samples, y_tensor_total)
    elif model_type_flag == 3:
        if model_flag == 3:
            spa_total_samples = HSI_create_pathes(spa_dr_data_torch, hsi_h, hsi_w, index, patch_length, 1)
        else:
            spa_total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 1)

        spe_total_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, index, patch_length, 2)
        torch_dataset_total = Data.TensorDataset(spa_total_samples, spe_total_samples, y_tensor_total)
    if last_batch_flag == 0:
        total_iter = Data.DataLoader(dataset=torch_dataset_total, batch_size=batch_size, shuffle=False, num_workers=0)
    elif last_batch_flag == 1:
        total_iter = Data.DataLoader(dataset=torch_dataset_total, batch_size=batch_size, shuffle=False, num_workers=0,
                                     drop_last=True)

    return total_iter

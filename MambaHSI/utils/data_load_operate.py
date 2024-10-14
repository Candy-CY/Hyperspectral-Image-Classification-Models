import os
import torch
import numpy as np
import scipy.io as sio
import torch.utils.data as Data
from sklearn import preprocessing
from sklearn.decomposition import PCA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def load_data(data_set_name, data_path='./data'):
    if data_set_name == 'UP':
        data = sio.loadmat(os.path.join(data_path, 'UP', 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'UP', 'PaviaU_gt.mat'))['paviaU_gt']
    elif data_set_name == 'Houston':
        data = sio.loadmat(os.path.join(data_path, 'Houston', 'Houston.mat'))['Houston']
        labels = sio.loadmat(os.path.join(data_path, 'Houston', 'Houston_GT.mat'))['Houston_GT']
    elif data_set_name == 'HongHu':
        data = sio.loadmat(os.path.join(data_path, 'HongHu', 'WHU_Hi_HongHu.mat'))['WHU_Hi_HongHu']
        labels = sio.loadmat(os.path.join(data_path, 'HongHu', 'WHU_Hi_HongHu_gt.mat'))['WHU_Hi_HongHu_gt']
    elif data_set_name == 'HanChuan':
        data = sio.loadmat(os.path.join(data_path, 'HanChuan', 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
        labels = sio.loadmat(os.path.join(data_path, 'HanChuan', 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']
    return data, labels


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


def HSI_create_pathes(data_padded, hsi_h, hsi_w, data_indexes, patch_length, flag, device='cuda:0'):
    h_p, w_p, c = data_padded.shape

    data_size = len(data_indexes)
    patch_size = patch_length * 2 + 1

    data_assign = index_assignment(data_indexes, hsi_h, hsi_w, patch_length)
    if flag == 1:
        # for spatial net data, HSI patch
        unit_data = np.zeros((data_size, patch_size, patch_size, c))
        unit_data_torch = torch.from_numpy(unit_data).type(torch.FloatTensor)
        for i in range(len(data_assign)):
            unit_data_torch[i] = select_patch(data_padded, data_assign[i][0], data_assign[i][1], patch_length)

    if flag == 2:
        # for spectral net data, HSI vector
        unit_data = np.zeros((data_size, c))
        unit_data_torch = torch.from_numpy(unit_data).type(torch.FloatTensor)
        for i in range(len(data_assign)):
            unit_data_torch[i] = select_vector(data_padded, data_assign[i][0], data_assign[i][1])

    return unit_data_torch


def get_aux_index(gt_reshape):
    aux_index =  np.where(gt_reshape == 0)[0]
    return aux_index


def generate_iter_1(data_padded, hsi_h, hsi_w, label_reshape, index, patch_length, batch_size, model_type_flag,
                    model_3D_spa_flag, last_batch_flag=0,data_auto_number=0, aa=0,bb=0,cc=0):
    # flag for single spatial net or single spectral net or spectral-spatial net
    # data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(device)
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

    elif model_type_flag >= 3:  # data for spectral-spatial net
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


    if last_batch_flag==1:
        train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                     drop_last=True)
        test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0,
                                drop_last=True)
        val_iter = Data.DataLoader(dataset=torch_dataset_val, batch_size=batch_size, shuffle=False, num_workers=0,
                                drop_last=True)
    else:

        train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
        val_iter = Data.DataLoader(dataset=torch_dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)
        test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, test_iter, val_iter


def generate_auxilary_iter(data_padded, hsi_h, hsi_w, label_reshape, aux_index, patch_length, batch_size, model_type_flag,
                    model_3D_spa_flag, last_batch_flag=0, data_auto_number=0, aa=0, bb=0, cc=0):
    # flag for single spatial net or single spectral net or spectral-spatial net

    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(device)
    # for data label
    aux_labels = label_reshape[aux_index] - 1
    y_tensor_aux = torch.from_numpy(aux_labels).type(torch.FloatTensor)

    # for data
    if model_type_flag == 1:  # data for single spatial net
        spa_aux_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, aux_index, patch_length, 1)
        if model_3D_spa_flag == 1:  # spatial 3D patch
            spa_aux_samples = spa_aux_samples.unsqueeze(1)
        torch_dataset_aux = Data.TensorDataset(spa_aux_samples, y_tensor_aux)


    elif model_type_flag == 2:  # data for single spectral net
        spe_aux_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, aux_index, patch_length, 2)
        torch_dataset_aux = Data.TensorDataset(spe_aux_samples, y_tensor_aux)

    elif model_type_flag >= 3:  # data for spectral-spatial net
        # spatail data
        spa_aux_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, aux_index, patch_length, 1)
        # spectral data
        spe_aux_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, aux_index, patch_length, 2)
        torch_dataset_aux = Data.TensorDataset(spa_aux_samples, spe_aux_samples, y_tensor_aux)

    if last_batch_flag == 1:
        aux_iter = Data.DataLoader(dataset=torch_dataset_aux, batch_size=batch_size, shuffle=True, num_workers=0,
                                     drop_last=True)
    else:
        aux_iter = Data.DataLoader(dataset=torch_dataset_aux, batch_size=batch_size, shuffle=True, num_workers=0)

    return aux_iter


def generate_image_iter(data_padded, hsi_h, hsi_w, label_reshape, index):
    def generate_label_map(num, hsi_w):
        num =np.array(num)
        idx_2d = np.zeros([num.shape[0], 2]).astype(int)
        idx_2d[:, 0] = num // hsi_w
        idx_2d[:, 1] = num % hsi_w
        label_map = np.zeros((hsi_h,hsi_w))
        for i in range(num.shape[0]):
            label_map[idx_2d[i, 0], idx_2d[i, 1]] = label_reshape[num[i]]
        return label_map.astype(int)

    # for data label
    train_labels = generate_label_map(index[0], hsi_w) - 1
    val_labels = generate_label_map(index[1], hsi_w) - 1
    test_labels = generate_label_map(index[2], hsi_w) - 1


    y_tensor_train = torch.from_numpy(train_labels).type(torch.FloatTensor)
    y_tensor_val = torch.from_numpy(val_labels).type(torch.FloatTensor)
    y_tensor_test = torch.from_numpy(test_labels).type(torch.FloatTensor)

    return y_tensor_train, y_tensor_val, y_tensor_test


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


def generate_all_iter(data, labels, patch_length,batch_size, device, model_type_flag, model_3D_spa_flag,all_index):
    hsi_h, hsi_w, channels = data.shape
    y_tensor_label = torch.from_numpy(labels).type(torch.FloatTensor)
    y_tensor_label = y_tensor_label[all_index]
    # assert labels.size == hsi_h*hsi_w
    data_padded = data_pad_zero(data, patch_length)
    # all_index = np.array([i for i in range(hsi_h*hsi_w)])
    data_padded_torch = torch.from_numpy(data_padded).type(torch.FloatTensor).to(device)
    if model_type_flag == 1:  # data for single spatial net
        spa_all_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, all_index, patch_length, 1,device)
        if model_3D_spa_flag == 1:  # spatial 3D patch
            spa_all_samples = spa_all_samples.unsqueeze(1)
        torch_dataset = Data.TensorDataset(spa_all_samples, y_tensor_label)
    elif model_type_flag == 2: # data for single spectral net
        spe_all_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, all_index, patch_length, 2,device)
        torch_dataset = Data.TensorDataset(spe_all_samples, y_tensor_label)
    elif model_type_flag == 3:  # data for spectral-spatial net
        spa_all_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, all_index, patch_length, 1,device)
        spe_all_samples = HSI_create_pathes(data_padded_torch, hsi_h, hsi_w, all_index, patch_length, 2,device)
        torch_dataset = Data.TensorDataset(spa_all_samples, spe_all_samples, y_tensor_label)
    all_iter = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return all_iter
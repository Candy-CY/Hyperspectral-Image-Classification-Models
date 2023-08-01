"""
Created on Wed Oct 21 21:10:24 2020

@author: xuegeeker
@blog: https://github.com/xuegeeker
@email: xuegeeker@163.com
"""

import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
import h5py
import torch.utils.data as Data
import datetime
import extract_samll_cubic


def load_dataset(Dataset):
    if Dataset == 'indian':
        mat_data = sio.loadmat('./datasets/Indian_pines_corrected.mat')
        mat_gt = sio.loadmat('./datasets/Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.97
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'pavia':
        uPavia = sio.loadmat('../datasets/PaviaU.mat')
        gt_uPavia = sio.loadmat('../datasets/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'paviaC':
        uPavia = sio.loadmat('../datasets/Pavia.mat')
        gt_uPavia = sio.loadmat('../datasets/Pavia_gt.mat')
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        data_hsi = data_hsi[:, -492:, :]
        gt_hsi = gt_hsi[:, -492:]
        TOTAL_SIZE = 148152
        VALIDATION_SPLIT = 0.999
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'salina':
        SV = sio.loadmat('./datasets/Salinas_corrected.mat')
        gt_SV = sio.loadmat('./datasets/Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'KSC':
        KSC = sio.loadmat('../datasets/KSC.mat')
        gt_KSC = sio.loadmat('../datasets/KSC_gt.mat')
        data_hsi = KSC['KSC']
        gt_hsi = gt_KSC['KSC_gt']
        TOTAL_SIZE = 5211
        VALIDATION_SPLIT = 0.95
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'BS':
        BS = sio.loadmat('../datasets/Botswana.mat')
        gt_BS = sio.loadmat('../datasets/Botswana_gt.mat')
        data_hsi = BS['Botswana']
        gt_hsi = gt_BS['Botswana_gt']
        TOTAL_SIZE = 3248
        VALIDATION_SPLIT = 0.99
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
        
    if Dataset == 'DFC2013':
        HS = sio.loadmat('../datasets/DFC2013_Houston.mat')
        gt_HS = sio.loadmat('../datasets/DFC2013_Houston_gt.mat')
        data_hsi = HS['DFC2013_Houston']
        gt_hsi = gt_HS['DFC2013_Houston_gt']
        TOTAL_SIZE = 15029
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
            
    if Dataset == 'DFC2018':
        DFC = sio.loadmat('../datasets/DFC2018_Houston.mat')
        gt_DFC = sio.loadmat('../datasets/DFC2018_Houston_gt.mat')
        data_hsi = DFC['DFC2018_Houston']
        gt_hsi = gt_DFC['DFC2018_Houston_gt']
        TOTAL_SIZE = 504712
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
        
    if Dataset == 'CK':
        f = h5py.File('../datasets/Chikusei.h5', 'r')
        data_hsi = f['X_train'][:]
        gt_hsi  = f['Y_train'][:]
        f.close()
        TOTAL_SIZE = 77592
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
        
    if Dataset == 'DN':
        DN = sio.loadmat('../datasets/Dioni.mat')
        gt_DN = sio.loadmat('../datasets/Dioni_gt.mat')
        data_hsi = DN['Dioni']
        gt_hsi = gt_DN['Dioni_gt']
        # f = h5py.File('../datasets/Dioni.h5', 'r')
        # data_hsi = f['X_train'][:]
        # gt_hsi  = f['Y_train'][:]
        # f.close()
        TOTAL_SIZE = 20024
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
        
    if Dataset == 'LK':
        f = h5py.File('../datasets/Loukia.h5', 'r')
        data_hsi = f['X_train'][:]
        gt_hsi  = f['Y_train'][:]
        f.close()
        TOTAL_SIZE = 13503
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT

def save_cmap(img, cmap, fname):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()

def sampling1(samples_num, ground_truth,flag):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        nb_val = samples_num
        #if proportion != 1:
        #    nb_val = max(int((1 - proportion) * len(indexes)), 3)
        #else:
        #    nb_val = 0
        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        if flag == 'indian':

            if samples_num == 20:

                if i + 1 == 7:
                    nb_val=10# 50
                elif i + 1 == 9:
                    nb_val=10  # 50

            if samples_num == 30:

                if i + 1 == 7:
                    nb_val=10# 50
                elif i + 1 == 9:
                    nb_val=10  # 50

            if samples_num == 50:
                if i + 1 == 1:
                    nb_val=26  # 50
                elif i + 1 == 7:
                    nb_val=16 # 50
                elif i + 1 == 9:
                    nb_val=11  # 50

            if samples_num == 80:
                if i + 1 == 1:
                    nb_val = 26  # 50
                elif i + 1 == 7:
                    nb_val = 16  # 50
                elif i + 1 == 9:
                    nb_val = 11  # 50
                elif i + 1 == 16:
                    nb_val = 60  # 50

            if samples_num == 100:
                if i + 1 == 1:
                    nb_val = 33  # 50
                elif i + 1 == 7:
                    nb_val = 20  # 50
                elif i + 1 == 9:
                    nb_val = 14  # 50
                elif i + 1 == 16:
                    nb_val = 75  # 50

            if samples_num == 150:
                if i + 1 == 1:
                    nb_val = 36  # 50
                elif i + 1 == 7:
                    nb_val = 22  # 50
                elif i + 1 == 9:
                    nb_val = 16  # 50
                elif i + 1 == 16:
                    nb_val = 80  # 50


            if samples_num == 200:
                if i + 1 == 1:
                    nb_val = 39  # 50
                elif i + 1 == 7:
                    nb_val = 24  # 50
                elif i + 1 == 9:
                    nb_val = 18  # 50
                elif i + 1 == 16:
                    nb_val = 85  # 50



        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

def sampling2(ground_truth, proportion):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

# choose all the samples
def sampling3(ground_truth, proportion):
    indexes = [j for j, x in enumerate(ground_truth.ravel().tolist())]
    return indexes

def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        '''
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 2:
            y[index] = np.array([100, 255, 100])/255.
        if item == 3:
            y[index] = np.array([0,0,255])/255.
        if item == 4:
            y[index] = np.array([255, 255, 0])/255.
        if item == 5:
            y[index] = np.array([255, 0, 255])/255.
        if item == 6:
            y[index] = np.array([255, 100, 100])/255.
        if item == 7:
            y[index] = np.array([150, 75, 255])/255.
        if item == 8:
            y[index] = np.array([150, 75, 75])/255.
        if item == 9:
            y[index] = np.array([100, 100, 255])/255.
        if item == 10:
            y[index] = np.array([0, 200, 200])/255.
        if item == 11:
            y[index] = np.array([0, 100, 100])/255.
        if item == 12:
            y[index] = np.array([100, 0, 100])/255.
        if item == 13:
            y[index] = np.array([128, 128, 0])/255.
        if item == 14:
            y[index] = np.array([200, 100, 0])/255.
        if item == 15:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 16:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 17:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 18:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 19:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 20:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 21:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 22:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 23:
            y[index] = np.array([0, 255, 0]) / 255.
        '''
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([255,182,193]) / 255.
        if item == 2:
            y[index] = np.array([60,179,113]) / 255.
        if item == 3:
            y[index] = np.array([255,165,0]) / 255.
        if item == 4:
            y[index] = np.array([65,105,225]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 6:
            y[index] = np.array([148,0,211]) / 255.
        if item == 7:
            y[index] = np.array([139,69,19]) / 255.
        if item == 8:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 9:
            y[index] = np.array([0,255,255])/255.
            #y[index] = np.array([0, 0, 0]) / 255.
        if item == 10:
            y[index] = np.array([128, 128, 0])/255.
        if item == 11:
            y[index] = np.array([255,255,0])/255.
        if item == 12:
            y[index] = np.array([121,255,49])/255.
        if item == 13:
            y[index] = np.array([255,49,183])/255.
        if item == 14:
            y[index] = np.array([112, 192, 188])/255.
        if item == 15:
            y[index] = np.array([183,121,121])/255.
        if item == 16:
            #y[index] = np.array([13,0,100])/255.
            y[index] = np.array([0, 0, 0]) / 255.

    return y

def generate_train_iter(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt, augmentation=False):
    
    y_train = gt[train_indices] - 1
    train_data = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)

    if augmentation == True:
        a = np.flip(x_train, 1)
        b = np.flip(x_train, 2)
        c = np.flip(b, 1)
        x_train = np.concatenate((a, b, c, x_train), 0)
        y_train = np.concatenate((y_train, y_train, y_train, y_train), 0)

    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )

    return train_iter

def generate_valida_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):

    y_test = gt[test_indices] - 1
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)
    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)

    valiada_iter = Data.DataLoader(
        dataset=torch_dataset_valida,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )

    return valiada_iter


def generate_test_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):
    
    y_test = gt[test_indices] - 1
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data,PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    #x_test = x_test_all[:-VAL_SIZE]
    #y_test = y_test[:-VAL_SIZE]

    x1_tensor_test = torch.from_numpy(x_test_all).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test)

    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )

    return test_iter

def generate_all_iter(TOTAL_SIZE, total_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):
    
    gt_all = gt[total_indices] - 1
    all_data = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)

    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )

    return all_iter

def generate_full_iter(whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt, FULL_SIZE, full_indices):

    gt_full = gt[full_indices] - 1
    full_data = extract_samll_cubic.select_small_cubic(FULL_SIZE, full_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    
    full_data.reshape(full_data.shape[0], full_data.shape[1], full_data.shape[2], INPUT_DIMENSION)
    full_tensor_data = torch.from_numpy(full_data).type(torch.FloatTensor).unsqueeze(1)
    full_tensor_data_label = torch.from_numpy(gt_full).type(torch.FloatTensor)
    torch_dataset_full = Data.TensorDataset(full_tensor_data, full_tensor_data_label)

    full_iter = Data.DataLoader(
        dataset=torch_dataset_full,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )

    return full_iter


def __Filter__(self, known):
    datas, targets = np.array(self.data), np.array(self.targets)
    mask, new_targets = [], []
    for i in range(len(targets)):
        if targets[i] in known:
            mask.append(i)
            new_targets.append(known.index(targets[i]))
    self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)


def generate_train_known_iter(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size,gt,known,augmentation=False):
    y_train = gt[train_indices] - 1
    train_data = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH,
                                                        padded_data, INPUT_DIMENSION)


    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)

    mask, new_targets = [], []
    for i in range(len(y_train)):
        if y_train[i] in known:
            mask.append(i)
            new_targets.append(known.index(y_train[i]))
            #new_targets.append(y_train[i])

    x_train, y_train = np.squeeze(np.take(x_train, mask, axis=0)), np.array(new_targets)

    if augmentation==True:
        a = np.flip(x_train,1)
        b = np.flip(x_train,2)
        c = np.flip(b, 1)
        x_train=np.concatenate((a,b,c,x_train),0)
        y_train=np.concatenate((y_train,y_train,y_train,y_train),0)

    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)

    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    return train_iter


def generate_test_known_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt,known):
    y_test = gt[test_indices] - 1
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data, PATCH_LENGTH, padded_data,
                                                       INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    # x_test = x_test_all[:-VAL_SIZE]
    # y_test = y_test[:-VAL_SIZE]
    mask, new_targets = [], []
    for i in range(len(y_test)):
        if y_test[i] in known:
            mask.append(i)
            new_targets.append(known.index(y_test[i]))
            #new_targets.append(y_test[i])

    x_test_all, y_test = np.squeeze(np.take(x_test_all, mask, axis=0)), np.array(new_targets)

    x1_tensor_test = torch.from_numpy(x_test_all).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, y1_tensor_test)

    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )

    return test_iter

def generate_test_unknown_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt,unknown,class_num):
    y_test = gt[test_indices] - 1
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data, PATCH_LENGTH, padded_data,
                                                       INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    # x_test = x_test_all[:-VAL_SIZE]
    # y_test = y_test[:-VAL_SIZE]
    mask, new_targets = [], []
    for i in range(len(y_test)):
        if y_test[i] in unknown:
            mask.append(i)
            #new_targets.append(known.index(y_test[i]))
            new_targets.append(class_num)
    x_test_all, y_test = np.squeeze(np.take(x_test_all, mask, axis=0)), np.array(new_targets)

    x1_tensor_test = torch.from_numpy(x_test_all).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, y1_tensor_test)

    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )

    return test_iter

def generate_fulltest_iter(TEST_SIZE, test_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt,known,class_num):
    y_test = gt[test_indices] - 1
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data, PATCH_LENGTH, padded_data,
                                                       INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    # x_test = x_test_all[:-VAL_SIZE]
    # y_test = y_test[:-VAL_SIZE]
    mask, new_targets = [], []
    for i in range(len(y_test)):
        if y_test[i] in known:
            #mask.append(i)
            new_targets.append(known.index(y_test[i]))
        else:
            new_targets.append(class_num)
    y_test =  np.array(new_targets)

    x1_tensor_test = torch.from_numpy(x_test_all).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, y1_tensor_test)

    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )

    return test_iter

def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):

    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    all_data = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data,
                                                      PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    train_data = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]
    # print('y_train', np.unique(y_train))
    # print('y_val', np.unique(y_val))
    # print('y_test', np.unique(y_test))
    # print(y_val)
    # print(y_test)

    # K.clear_session()  # clear session before next loop

    # print(y1_train)
    #y1_train = to_categorical(y1_train)  # to one-hot labels
    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test)

    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)


    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    valiada_iter = Data.DataLoader(
        dataset=torch_dataset_valida,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    return train_iter, valiada_iter, test_iter, all_iter #, y_test


def generate_png(gt_hsi,pred_test,flag,h,w,num):


    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)

    for i in range(len(pred_test)):
        pred_test[i] = pred_test[i] + 1

    for i in range(len(gt)):
        if gt[i] == 255.0:
            gt[i] = 0.0
        else:
            gt[i] +=1.0

    y_list = list_to_colormap(pred_test)
    y_gt = list_to_colormap(gt)


    y_re = np.reshape(y_list, (h, w, 3))
    gt_re = np.reshape(y_gt, (h, w, 3))

    day = datetime.datetime.now()
    day_str = day.strftime('%m_%d_%H_%M')

    #path = './maps/'+str(flag)+'/'
    path = './maps/' + str(flag)
    classification_map(y_re, gt_re, 600,
                       path + '_' + 'Time_'+str(day_str)+'_'+str(flag)+'_'+str(num)+'num.eps')
    #classification_map(gt_re, gt_re, 600,
    #                   path + 'Time_gt'+str(day_str)+'_'+str(flag)+'.eps')
    print('------Get classification maps successful-------')


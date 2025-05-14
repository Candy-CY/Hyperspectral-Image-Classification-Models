# -*- coding: utf-8 -*-
"""
@author: mengxue.zhang
lazy load HSI patches (need remarkably less MEMORY whatever window size is)
"""

import scipy.io as sio
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import h5py
from keras.utils import np_utils

## set global seed to make sure reproducibility
seed = 666
random_state = np.random.RandomState(seed=seed)
default_max_hw = 35

## global constant value
data_name_dict = {'1': 'PaviaU',
             '2':'Indian_pines',
             '3':'Houston'}

train_num_dict = {
    '1': [332, 932, 105, 153, 67, 251, 67, 184, 47],
    '2': [5, 143, 83, 24, 48, 73, 3, 48, 2, 97, 246, 59, 21, 127, 39, 9],
    '3':[50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
                  }

# disjoint_train_num_dict = {
#     '1': [514, 540, 392, 231, 375, 532, 265, 548, 524],
#     '2': [762, 435, 232, 394, 235, 470, 1424, 328, 728, 29,146,16,10,132,291,57],
#     '3': [198, 190, 192, 188, 186, 182, 196, 191, 193, 191, 181, 192, 184, 181, 187],
#                   }

image_size_dict = {
    '1':[610, 340, 103],
    '2':[145, 145, 200],
    '3':[349, 1905, 144],
                   }

## the last two number control the picture effectiness
fc_dict = {
    '1':[56, 33, 13, 1.0, 0.5],
    '2':[50, 27, 17, 1.0, 1.0],
    '3':[59, 40, 23, 1.0, 0.5],
                   }

class_name_dict = {
    '1':['Asphalt','Meadows','Gravel','Trees','Painted metal sheets',
         'Bare soil','Bitumen','Self-Blocking Bricks','Shadows'],
    '2': ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
          'Grass-trees', 'Grass-pasture-moved', 'Hay-windrowed', 'Oats',
          'Soybeans-notill', 'Soybeans-mintill', 'Soybeans-clean', 'Wheat',
          'Woods', 'Bldg-grass-tree-drivers', 'Stone-steel-towers'],
    '3':['Healthy grass','Stressed grass','Synthetic grass','Trees',
         'Soil','Water','Residential','Commercial','Road','Highway',
         'Railway','Parking Lot1','Parking Lot2','Tennis court','Running track'],
}


color_map_dict = {
    '1': np.array([[0, 0, 255],
                            [76, 230, 0],
                            [255, 190, 232],
                            [255, 0, 0],
                            [156, 156, 156],
                            [255, 255, 115],
                            [0, 255, 197],
                            [132, 0, 168],
                            [0, 0, 0]]),
    '2':np.array([[0, 168, 132],
                            [76, 0, 115],
                            [0, 0, 0],
                            [190, 255, 232],
                            [255, 0, 0],
                            [115, 0, 0],
                            [205, 205, 102],
                            [137, 90, 68],
                            [215, 158, 158],
                            [255, 115, 223],
                            [0, 0, 255],
                            [156, 156, 156],
                            [115, 223, 255],
                            [0, 255, 0],
                            [255, 255, 0],
                            [255, 170, 0]]),

    '3':np.array([[0, 168, 132],
                            [76, 0, 115],
                            [0, 0, 0],
                            [190, 255, 232],
                            [255, 0, 0],
                            [115, 0, 0],
                            [205, 205, 102],
                            [137, 90, 68],
                            [215, 158, 158],
                            [255, 115, 223],
                            [0, 0, 255],
                            [156, 156, 156],
                            [115, 223, 255],
                            [0, 255, 0],
                            [255, 255, 0]]),}

data_path_dir = {
    '1':[['Dataset/PaviaU.mat', 'x'],
         ['Dataset/PaviaU_gt.mat', 'y']],
    '2': [['Dataset/Indian_pines_corrected.mat', 'x'],
          ['Dataset/Indian_pines_gt.mat', 'y']],
    '3': [['Dataset/Houston.mat', 'x'],
          ['Dataset/Houston_gt.mat', 'y']],
}

disjoint_data_path_dir = {
    '1': [['Dataset/PaviaU.mat', 'x'],
          ['Dataset/DS_PaviaU_gt.mat', 'y'],
          ['Dataset/DS_PaviaU_gt2.mat', 'y'],],
    '2': [['Dataset/Indian_pines_corrected.mat', 'x'],
          ['Dataset/DS_Indian_pines_gt.mat', 'y'],
          ['Dataset/DS_Indian_pines_gt2.mat', 'y']],
    '3': [['Dataset/Houston.mat', 'x'],
           ['Dataset/Houston_train_gt.mat', 'y'],
          ['Dataset/Houston_test_gt.mat', 'y']],
}


def try_load_data(path1, str1):
    try:
        data = sio.loadmat(path1)
        X = data[str1]
    except Exception:
        data = h5py.File(path1,'r')
        X = data[str1][:]
        if len(X.shape) == 3:
            X = np.transpose(X, axes=[2, 1, 0])
        else:
            X = np.transpose(X, axes=[1, 0])
    return X


def load_data(dataID=1, fixed=False):
    if fixed:
        data_path = disjoint_data_path_dir[str (dataID)]
    else:
        data_path = data_path_dir[str(dataID)]

    X = try_load_data(data_path[0][0], data_path[0][1])

    if fixed:
        Y_train = try_load_data(data_path[1][0], data_path[1][1])
        Y_test = try_load_data(data_path[2][0], data_path[2][1])
        return X, [Y_train, Y_test]
    else:
        Y = try_load_data(data_path[1][0], data_path[1][1])
        return X, Y


def featureNormalize(X,type, eps=0.0):
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm

    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX+eps)
        return X_norm

    elif type==3:
        sigma = np.std(X,0)
        X_norm = X/sigma
        return X_norm


def mat2rgb(mat, eps=0.0):
    sz = np.shape(mat)
    if len(sz) == 3:
        r = np.reshape(mat[:,:,0], [sz[0]*sz[1]])
        r = np.expand_dims(np.reshape(featureNormalize(r, type=2, eps=eps), [sz[0], sz[1]]), axis=-1)
        g = np.reshape(mat[:,:,1], [sz[0]*sz[1]])
        g = np.expand_dims(np.reshape(featureNormalize(g, type=2, eps=eps), [sz[0], sz[1]]), axis=-1)
        b = np.reshape(mat[:,:,2], [sz[0]*sz[1]])
        b = np.expand_dims(np.reshape(featureNormalize(b, type=2, eps=eps), [sz[0], sz[1]]), axis=-1)
        rgb = np.concatenate([r, g, b], axis=-1)
        return rgb
    else:
        gray = np.reshape(mat[:,:], [sz[0]*sz[1]])
        gray = np.reshape(featureNormalize(gray, type=2, eps=eps), [sz[0], sz[1]])
        return gray


def H_LazyProcessing(dataID, num_PC, w=11, fixed=False):
    hw = w // 2
    X, Y = load_data(dataID, fixed=fixed)
    [row, col, n_feature] = X.shape

    X_PCAMirrow, rgb = PCAMirrowCut(dataID, X, hw=hw, num_PC=num_PC)

    if fixed:
        Y1 = Y[0]
        Y1 = Y1.reshape(row * col, 1)

        Y2 = Y[1]
        Y2 = Y2.reshape(row * col, 1)

        Y = [Y1, Y2]
    else:
        Y = Y.reshape(row * col, 1)

    return X_PCAMirrow, rgb, Y, [row, col, n_feature]


def LazyProcessing(dataID, num_PC, w=11, fixed=False, type='HSI'):
    # if type == 'HSI':
    return H_LazyProcessing(dataID, num_PC, w, fixed)


def get_path(dataID, name='_pca_u'):
    file_path = './temp_vars/' + data_name_dict[str(dataID)] + name + '.npy'
    return file_path


def dimensionReduction2d(x, num=3, shape=[], type='pca'):
    def pca(x, n_components, shape):
        shp = shape
        data = np.transpose(x, [1, 0])
        data_norm = data - np.mean(data, 1, keepdims=True)

        sigma = np.cov(data_norm)
        [U, S, V] = np.linalg.svd(sigma)

        u = U[:, 0:n_components]
        s = S[0:n_components]
        v = V[0:n_components, :]

        # project to a new column vector space
        data_pca = np.dot(np.transpose(u), data_norm)

        # rescale each variable to unit variance.
        epison = 0.0
        data_pca = np.dot(np.diag((1 / (np.sqrt(s + epison)))), data_pca)
        data_pca = np.transpose(data_pca, [1, 0])
        return data_pca.astype(dtype=np.float32)

    return pca(x, num, shape=shape)


def PCAMirrowCut(dataID, X, hw, num_PC=0, type='HSI'):
    cnum = image_size_dict[str(dataID)][2]

    file_path = get_path(dataID, '_mirror_pca')
    rgb_path = get_path(dataID, '_rgb')

    if os.path.exists(file_path) and num_PC != 0:
        X_extension = np.load(file_path)
        X_extension = X_extension[:, :, :num_PC]
        rgb_extension = np.load(rgb_path)
    else:
        [row, col, n_feature] = X.shape
        X = X.reshape(row * col, n_feature)

        X_rgb = gen_false_color(dataID)

        if num_PC != 0:
            X = dimensionReduction2d(X, cnum, shape=[row, col, n_feature], type='pca')
            X = X.reshape(row, col, X.shape[-1])
        else:
            X = X.reshape(row, col, cnum)

        X_extension = mirror_concatenate(X)

        rgb_extension = mirror_concatenate(X_rgb)

        if num_PC != 0:
            np.save(file_path, X_extension)
            np.save(rgb_path, rgb_extension)
            X_extension = X_extension[:,:,0:num_PC]

    b = default_max_hw - hw
    rgb_extension = rgb_extension[b:-b, b:-b, :]
    X_extension = X_extension[b:-b, b:-b, :]

    return X_extension, rgb_extension

def mirror_concatenate(x, max_hw=default_max_hw):
    x_extension = cv2.copyMakeBorder(x, max_hw, max_hw, max_hw, max_hw, cv2.BORDER_REFLECT)
    return x_extension

def generate_whole_batch(dataID=1, num_PC=4, w=11, batch_size=64):
    X_PCAMirrow, rgb,  Y, shape_list = LazyProcessing(dataID, num_PC, w=w)

    row = shape_list[0]
    col = shape_list[1]

    K = row * col

    n_class = Y[-1].max()

    hw = w // 2

    steps = math.ceil(K / batch_size)

    def generate_batch():
        while True:
            for i in range(0, K, batch_size):
                idx = np.arange(i, min(K, i + batch_size))
                index_row = np.ceil((idx + 1) * 1.0 / col).astype(np.int32)
                index_col = (idx + 1) - (index_row - 1) * col

                index_row += hw - 1
                index_col += hw - 1

                patches = []
                for j in range(idx.size):
                    a = index_row[j] - hw
                    b = index_col[j] - hw
                    patch = X_PCAMirrow[a:a + w, b:b + w, :]
                    patches.append(patch)

                patches = np.array(patches)

                yield patches

    return generate_batch(), steps

def elimate_unlabeled_pixel(aMap, dataID, fixed=False):
    X, Y = load_data(dataID=dataID,fixed=fixed)
    if len(Y) == 2:
        Y = Y[-1]
    [row,col,n_feature] = X.shape
    K = row * col
    if len(aMap.shape)==2:
        aMap = aMap.reshape(K, 1)

    Y = Y.reshape(K, 1)
    aMap[np.where(Y == 0)[0]] = -1
    return aMap

def get_class_num(dataID):
    return len(train_num_dict[str(dataID)])

def gen_false_color(dataID=1):
    [X, _] = load_data(dataID=dataID)

    rgb = fc_dict[str(dataID)]
    sz = image_size_dict[str(dataID)]

    X = X[:,:, rgb[0:3]]
    x = mat2rgb(X)
    x = np.power(rgb[3] * x, rgb[4])

    return x
def get_images(batchs, step):
    X = []
    idx = 0
    try:
        for images, label in batchs:
            if type(images) == list:
                images = images[0]
            idx = idx + 1

            X.append(images)
            if idx == step:
                break

    except ValueError:
        for images in batchs:
            if type(images) == list:
                images = images[0]
            idx = idx + 1
            X.append(images)
            if idx == step:
                break
    return np.concatenate(X, axis=0)

def get_labels(batchs, step=100, argmax=True):
    Y = []
    idx = 0

    for images, label in batchs:
        if type(label) == list:
            label = label[0]
        idx = idx + 1

        Y.extend(label)
        if idx == step:
            break
    if argmax:
        return np.argmax(np.array(Y),1)
    else:
        return np.array(Y)


def generate_fixed_train_test_batch(dataID=1, num_PC=4, w=11, batch_size=64):

    X_PCAMirrow, rgb, Y, shape_list = LazyProcessing(dataID, num_PC, w=w, fixed=True)
    Y_train = Y[0]
    Y_test = Y[1]

    row = shape_list[0]
    col = shape_list[1]

    n_class = Y_test.max()

    hw = w // 2

    train_idx = list()
    test_idx = list()

    for i in range(1, n_class + 1):
        train_i = np.where(Y_train == i)[0]
        test_i = np.where(Y_test == i)[0]

        train_idx.extend(train_i)
        test_idx.extend(test_i)

    # when the sample is limit, use full batch training
    if len(train_idx) < 1 * batch_size:
        train_batch_size = len(train_idx)
    else:
        train_batch_size = batch_size

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    train_step = math.ceil(train_idx.size / train_batch_size)
    test_step = math.ceil(test_idx.size / batch_size)

    def generate_batch(idx, X_PCAMirrow, Y, batch_size, shuffle=False):

        num = idx.size
        hw = w // 2
        nclass = Y[idx].max()

        while True:
            if shuffle:
                random_state.shuffle(idx)

            for i in range(0, num, batch_size):
                bi = idx[np.arange(i, min(num, i + batch_size))]
                index_row = np.ceil((bi + 1) * 1.0 / col).astype(np.int32)
                index_col = (bi + 1) - (index_row - 1) * col
                index_row += hw - 1
                index_col += hw - 1

                patches = []
                for j in range(bi.size):
                    a = index_row[j] - hw
                    b = index_col[j] - hw
                    patch = X_PCAMirrow[a:a + w, b:b + w, :]
                    patches.append(patch)
                patches = np.array(patches)

                labels = Y[bi, :] - 1

                yield [patches], \
                      [np_utils.to_categorical(labels[:,0], Y.max())]

    return (generate_batch(train_idx, X_PCAMirrow, Y_train, train_batch_size, shuffle=True), train_step, generate_batch(test_idx, X_PCAMirrow, Y_test, batch_size), test_step)


def generate_train_test_batch(dataID=1, num_list=[], num_PC=4, w=11, batch_size=64, shuffle=True):

    if not len(num_list):
        num_list = train_num_dict[str(dataID)]

    train_num_all = sum(num_list)

    X_PCAMirrow, rgb, Y, shape_list = LazyProcessing(dataID, num_PC, w=w)

    row = shape_list[0]
    col = shape_list[1]

    n_class = Y.max()

    hw = w // 2

    train_idx = list()
    test_idx = list()

    for i in range(1, n_class + 1):
        index = np.where(Y == i)[0]
        n_data = index.shape[0]
        random_state.shuffle(index)
        train_num = num_list[i - 1]

        train_idx.extend(index[0:train_num])
        test_idx.extend(index[train_num:n_data])

    # when the sample is limit, use full batch training
    if len(train_idx) < 1 * batch_size:
        train_batch_size = len(train_idx)
    else:
        train_batch_size = batch_size

    train_idx = np.array(train_idx)
    if len(test_idx) == 0:
        test_idx = train_idx
    else:
        test_idx = np.array(test_idx)

    train_step = math.ceil(train_idx.size / train_batch_size)
    test_step = math.ceil(test_idx.size / batch_size)

    def generate_batch(idx, X_PCAMirrow, Y, batch_size, shuffle=False):

        num = idx.size
        hw = w // 2
        nclass = Y[idx].max()

        while True:
            if shuffle:
                random_state.shuffle(idx)

            for i in range(0, num, batch_size):
                bi = idx[np.arange(i, min(num, i + batch_size))]
                index_row = np.ceil((bi + 1) * 1.0 / col).astype(np.int32)
                index_col = (bi + 1) - (index_row - 1) * col
                index_row += hw - 1
                index_col += hw - 1

                patches = []
                for j in range(bi.size):
                    a = index_row[j] - hw
                    b = index_col[j] - hw
                    patch = X_PCAMirrow[a:a + w, b:b + w, :]
                    patches.append(patch)
                patches = np.array(patches)

                # for j in range(bi.size):
                #     a = index_row[j] - hw
                #     b = index_col[j] - hw
                #     rgb_patch = rgb[a:a + w, b:b + w, :]
                #     l = Y[bi[j], :]
                #     plt.imsave(fname='./fm/class_'+str(int(j))+'.png', arr=mat2rgb(rgb_patch))

                labels = Y[bi, :] - 1

                yield [patches], \
                      [np_utils.to_categorical(labels[:,0], Y.max())]#, np_utils.to_categorical(labels[:,0], Y.max())]
    # shuffle=True
    return (generate_batch(train_idx, X_PCAMirrow, Y, train_batch_size, shuffle=shuffle), train_step, generate_batch(test_idx, X_PCAMirrow, Y, batch_size), test_step)


def draw_result(labels, probs=None, dataID=1, border=False):

    num_class = labels.max() + 1
    row = image_size_dict[str(dataID)][0]
    col = image_size_dict[str(dataID)][1]

    palette = color_map_dict[str(dataID)]
    palette = palette * 1.0 / 255

    X_result = np.zeros((labels.shape[0], 3))

    X_result[np.where(labels == -1), 0] = 255 * 1.0 / 255
    X_result[np.where(labels == -1), 1] = 255 * 1.0 / 255
    X_result[np.where(labels == -1), 2] = 255 * 1.0 / 255

    t = palette[1, 0]
    for i in range(0, num_class):
        X_result[np.where(labels == i), 0] = palette[i, 0]
        X_result[np.where(labels == i), 1] = palette[i, 1]
        X_result[np.where(labels == i), 2] = palette[i, 2]

    X_result[np.where(labels == -1), 0] = 255 * 1.0 / 255
    X_result[np.where(labels == -1), 1] = 255 * 1.0 / 255
    X_result[np.where(labels == -1), 2] = 255 * 1.0 / 255

    X_result = np.reshape(X_result, (row, col, 3))
    if border:
        new_X_result = np.zeros([row+2, col+2, 3])
        new_X_result[1:-1, 1:-1, :] =  X_result
        X_result = new_X_result

    plt.axis("off")
    plt.imshow(X_result)
    return X_result

def draw_bar(dataID=1):
    bar_w = 0.1
    bar_h = 0.05

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')

    palette = color_map_dict[str(dataID)] * 1.0 / 255
    cname = class_name_dict[str(dataID)]

    l = np.shape(palette)[0]
    for idx in range(l):
        i = l - idx - 1
        c = palette[i, :]
        rect = patches.Rectangle((0, bar_h * idx), bar_w, bar_h, color=c)
        ax1.add_patch(rect)

        cn = cname[i]
        plt.text(bar_w * 1.2, bar_h * idx + bar_h / 8, cn, fontsize=16)

    plt.axis('off')
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

    plt.xlim(xmin=0, xmax=bar_w * 3)
    plt.ylim(ymin=0, ymax=bar_h * l)
    fig1.savefig('./' + data_name_dict[str(dataID)] + '_bar.svg', format='svg', bbox_inches='tight', pad_inches=0.0)


def draw_gt(dataID=1, background=[0,0,0], bar=False, fixed=False):
    [X, Y] = load_data(dataID=dataID, fixed=fixed)

    if fixed:
        Y_train = Y[0].astype(np.int8) - 1
        map = elimate_unlabeled_pixel(Y_train, dataID=dataID)
        X_result = draw_result(map, probs=None, dataID=dataID, border=True)
        plt.imsave('./' + data_name_dict[str(dataID)] + '_train_gt.svg', X_result)

        Y_test = Y[1].astype(np.int8) - 1
        map = elimate_unlabeled_pixel(Y_test, dataID=dataID, fixed=True)
        X_result = draw_result(map, probs=None, dataID=dataID, border=True)
        plt.imsave('./' + data_name_dict[str(dataID)] + '_test_gt.svg', X_result)
    else:
        Y = Y.astype(np.int8) - 1
        map = elimate_unlabeled_pixel(Y, dataID=dataID)
        X_result = draw_result(map, probs=None, dataID=dataID, border=True)
        plt.imsave('./' + data_name_dict[str(dataID)] + '_gt.svg', X_result)

    if bar:
        draw_bar(dataID=dataID)

def draw_false_color(dataID=1):
    X = gen_false_color(dataID=dataID)
    plt.imsave('./' + data_name_dict[str(dataID)] + '_rgb.svg', X)

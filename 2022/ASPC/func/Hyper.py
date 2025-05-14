# -*- coding: utf-8 -*-
"""
@author: Zhu
"""
import numpy as np


def sys_random_fixed(net_seed=2):
    import random, os, torch
    torch.manual_seed(net_seed)  # cpu
    torch.cuda.manual_seed(net_seed)  # gpu
    torch.cuda.manual_seed_all(net_seed)  # gpu
    np.random.seed(net_seed)  # numpy
    random.seed(net_seed)  # numpy
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(net_seed)


def calcAccuracy(predict, label):
    """
    功能：计算predict相当于label的正确率
    输入：（预测值，真实值）
    输出：正确率
    备注：输入均为一维数据，此时标签已经减一处理
    """
    num = len(label)
    numClass = np.max(label) + 1
    OA = np.sum(predict == label) * 1.0 / num
    correct_sum = np.zeros(numClass)
    reali = np.zeros(numClass)
    predicti = np.zeros(numClass)
    producerA = np.zeros(numClass)
    for i in range(0, numClass):  # 对于每个类别 0-8
        correct_sum[i] = np.sum(label[np.where(predict == i)] == i)  # 该类别预测对的数目
        reali[i] = np.sum(label == i)  # 该类别真正的数目
        predicti[i] = np.sum(predict == i)  # 该类别预测的数目
        producerA[i] = correct_sum[i] / reali[i]  # 该类别的正确率
    Kappa = (num * np.sum(correct_sum) - np.sum(reali * predicti)) * 1.0 / (
            num * num - np.sum(reali * predicti))  # 计算出某综合参数
    print('OA:', OA)
    print('AA:', producerA.mean(), 'for each part:', producerA)
    print('Kappa:', Kappa)
    # for i in producerA:
    #     print("%.3f" % i)
    # print("%.3f" % OA)
    # print("%.3f" % producerA.mean())
    # print("%.3f" % Kappa)
    return OA, Kappa, producerA


def create_patch(zeroPaddedX, row_index, col_index, patchSize=5):
    """
    功能：对指定像素切割一次patch
    输入：（零增广数据，行坐标，列坐标，patch大小）
    输出：该指定像素以及周围数据所构成的patch
    备注：返回如5*5*103型数据
    """
    row_slice = slice(row_index, row_index + patchSize)
    col_slice = slice(col_index, col_index + patchSize)
    patch = zeroPaddedX[row_slice, col_slice, :]
    return np.asarray(patch).astype(np.float32)


def dataEnrich(X, y):
    """
    功能：对数据扩充增强
    输入：（原始数据X，标签数据y）
    输出：（扩充后的数据X，扩充后的标签数据y）
    备注：随机对patch旋转角度而更充分地利用数据,5*5*103
    """
    from scipy.ndimage.interpolation import rotate
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts
    # repeat for every label and concat
    newX = X[y == uniqueLabels[0], :, :, :].repeat(round(labelInverseRatios[0]), axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y == label, :, :, :].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :, :, :]
    newY = newY[rand_perm]
    # random flip each patch
    for i in range(int(newX.shape[0] / 2)):
        patch = newX[i, :, :, :]
        num = np.random.randint(0, 3)
        if num == 0:
            flipped_patch = np.flipud(patch)  # 矩阵上下翻转函数
        if num == 1:
            flipped_patch = np.fliplr(patch)  # 矩阵左右翻转函数
        if num == 2:
            no = (np.random.randint(12) - 6) * 30
            flipped_patch = rotate(patch, no, axes=(1, 0),  # 矩阵旋转函数
                                   reshape=False, output=None, order=3, mode='constant',
                                   cval=0.0, prefilter=False)
        newX[i, :, :, :] = flipped_patch  # 替换随机翻转后的数据
    return newX, newY


def dataMess(X_train, y_train):
    """
    功能：按照随机顺序打乱两个patch序列
    输入：（原始数据X，原始标签y）
    输出：打乱后的两个序列
    备注：无
    """
    x_trains_alea_indexs = list(range(len(X_train)))  # 生成一串序列
    np.random.shuffle(x_trains_alea_indexs)  # 对该序列进行乱序
    X_train = X_train[x_trains_alea_indexs]  # 根据该序列打乱训练序列
    y_train = y_train[x_trains_alea_indexs]  # 根据该序列打乱标记值
    return X_train, y_train


def dataNormalize(X, type=1):
    """
    功能：数据归一化
    输入：（原始数据，归一化类型）
    输出：归一化后数据
    备注：type==1最常用;与PCA二选一
        type==0 x = (x-mean)/std(x) #标准化
        type==1 x = (x-min(x))/(max(x)-min(x)) #归一化
        type==2 x = (2x-max(x))/(max(x))
    """
    if type == 0:  # 均值0，max、min不定
        mu = np.mean(X)
        X_norm = X - mu
        sigma = np.std(X_norm)
        X_norm = X_norm / sigma
        return X_norm
    elif type == 1:  # [0,1],最常用
        minX = np.min(X)
        maxX = np.max(X)
        X_norm = X - minX
        X_norm = X_norm / (maxX - minX)
    elif type == 2:  # 均值非零，[-1,1]
        maxX = np.max(X)
        X_norm = 2 * X - maxX
        X_norm = X_norm / maxX
    return X_norm.astype(np.float32)


def displayClassTable(n_list, matTitle=""):
    """
    功能：打印list的各元素
    输入：（list）
    输出：无
    备注：无
    """
    from pandas import DataFrame
    print("\n+--------- 原始输入数据" + matTitle + "统计结果 ------------+")
    lenth = len(n_list)  # 一共n个分类
    column = range(1, lenth + 1)
    table = {'Class': column, 'Total': [int(i) for i in n_list]}
    table_df = DataFrame(table).to_string(index=False)
    print(table_df)
    print('All available data total ' + str(int(sum(n_list))))
    print("+---------------------------------------------------+")


def kMeans(dataSet, k):
    """
    函数说明：k-mean聚类算法
    Parameter：
        dataSet：数据集，格式如 10000*2
        k:分类类别数
    Return：
        k个质心坐标（格式如10*2）、样本的分配结果（格式如10000*1）
    """
    from sklearn.cluster import KMeans
    labels = KMeans(k, init='k-means++', n_init=k, max_iter=3000).fit(dataSet).labels_
    # KMeans(16,init='k-means++',n_init=16,max_iter=3000).fit(test_gt_pred).labels_
    return labels


def listClassification(Y, matTitle=''):
    """
    功能：对标签数据计数并打印
    输入：（原始标签数据，是否打印）
    输出：分类结果
    备注：无
    """
    numClass = np.max(Y)  # 获取分类数
    listClass = []  # 用列表依次存储各类别的数量
    for i in range(numClass):
        listClass.append(len(np.where(Y == (i + 1))[0]))
    displayClassTable(listClass, matTitle)
    return listClass


def matDataLoad(filename='./datasets/Salinas.mat'):
    """
    功能：读取.mat文件的有效数据
    输入：（文件名字）
    输出：数据
    备注：无
    """
    try:  # 读取v5类型mat数据
        import scipy.io
        data = scipy.io.loadmat(filename)
        x = list(data.keys())
        data = data[x[-1]]  # 读取出有用数据
    except:  # 读取v7类型mat数据
        import h5py
        data = h5py.File(filename, 'r')
        x = list(data.keys())
        data = data[x[1]][()].transpose(2, 1, 0)  # 调整为（row,col,band）
    if 'gt' in filename:
        return data.astype('long')  # 标签数据输出为整型
    else:
        return data.astype('float32')  # 原始数据输出为浮点型


def padWithZeros(X, patchSize=5):
    """
    功能：对数据进行零增广
    输入：（原始数据，patch大小）
    输出：零增广后数据
    备注：无
    """
    zeroSize = int((patchSize - 1) / 2)  # 零增广个数
    zeroPaddedX = np.zeros((X.shape[0] + 2 * zeroSize, X.shape[1] + 2 * zeroSize, X.shape[2]))
    x_offset = zeroSize
    y_offset = zeroSize
    zeroPaddedX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return zeroPaddedX.astype(np.float32)


def patchesCreate(X, y, patchSize=9):
    """
    功能：将数据挨像素切割成patch
    输入：（原始数据X，原始标签数据y，patch大小）
    输出：分patch之后的数据与标签数据并打印
    备注：分割前是“块”，分割后是“条”
    """
    numClass = np.max(y)
    row, col = y.shape
    zeroPaddedX = padWithZeros(X, patchSize)
    nb_samples = np.zeros(numClass)  # 用于统计每个类别的数据
    patchesData = []  # 用于收集有效像素的patch
    patchesLabels = []  # 收集patch对应的标签
    for i in range(row):
        for j in range(col):  # 对于610*340这么多每个像素的1*103数据
            label = y[i, j]  # 提取对应像素的标注值，若为零则为背景，应该剔除掉
            if label > 0:  # 仅保留有用标注
                patch = create_patch(zeroPaddedX, i, j, patchSize)  # 5*5*103，以该像素为中心切取周围两圈构成5*5
                nb_samples[label - 1] += 1  # 统计每种类型各自的数量
                patchesData.append(patch.astype(np.float32))  # 保存每种类型的数据
                patchesLabels.append(label)  # 保存该patch的标签
    displayClassTable(nb_samples)  # 打印每种类型对应的数量
    return np.array(patchesData), np.array(patchesLabels)


def patchesCreate_all(X, patchSize=9):
    """
    功能：将数据挨像素切割成patch(包括背景部分)
    输入：（原始数据X，patch大小）
    输出：分patch之后的数据
    备注：不输入标签数据
    """
    row, col, layers = X.shape
    zeroPaddedX = padWithZeros(X, patchSize)
    patchesData = []  # 用于收集有效像素的patch
    for i in range(row):
        for j in range(col):  # 对于610*340这么多每个像素的1*103数据
            patch = create_patch(zeroPaddedX, i, j, patchSize)  # 5*5*103，以该像素为中心切取周围两圈构成5*5
            patchesData.append(patch.astype(np.float32))  # 保存每种类型的数据
    return np.array(patchesData).astype(np.float32)


def patchesCreate_balance(X, y, patchSize=5, train_mode=False):
    """
    功能：重构二分类高光谱数据集,使两者数量平衡
    输入：（原始数据X，原始数据y,patch大小,是否为训练模式）
    输出：分patch之后的数据及其标签
    备注：若为测试模式则全部取出,不进行平衡
    """
    row, col = y.shape
    zeroPaddedX = padWithZeros(X, patchSize)
    nb_samples = np.zeros(2)  # 用于统计每个类别的数据
    patchesData = []  # 用于收集有效像素的patch
    patchesLabels = []  # 收集patch对应的标签
    if train_mode:
        random_choose = np.random.randint(15, size=(row, col))
    else:
        random_choose = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            label = y[i, j]
            if label > 0 or random_choose[i, j] == 0:  # 由于背景过多,故降低背景样本数量
                patch = create_patch(zeroPaddedX, i, j, patchSize)
                nb_samples[label] += 1  # 统计每种类型各自的数量
                patchesData.append(patch.astype(np.float32))  # 保存每种类型的数据
                patchesLabels.append(label)  # 保存该patch的标签
    displayClassTable(nb_samples)  # 打印每种类型对应的数量
    return np.ascontiguousarray(patchesData).astype('float32'), np.ascontiguousarray(patchesLabels).astype('long')


def PCANorm(X, numPCA=30):
    """
    功能：PCA降维
    输入：（原始数据，降维后维度）
    输出：降维后数据
    备注：输入输出都为三维数据；归一化与PCA取其一
    """
    from sklearn.decomposition import PCA  # PCA降维
    newX = np.reshape(X, (-1, X.shape[2]))  # 将空间信息铺开
    pca = PCA(n_components=numPCA, whiten=True)  # 定义PCA信息
    newX = pca.fit_transform(newX)  # 降维操作
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numPCA))  # 整形回原来的空间形状
    return newX


def print_f(data='', path='./records/', fileName='data', show=True):
    """
    功能：对数据既打印又保存到.txt文件到本地
    输入：（打印数据，保存为的文件名）
    输出：无
    备注：无
    """
    import os
    os.makedirs(path, exist_ok=True)
    with open(path + fileName + '.txt', 'a') as f:  # 'a'表示append,即在原来文件内容后继续写数据（不清除原有数据）
        f.write(str(data) + "\n")
    if show:
        print(data)


def one_hot_encoding(Y):
    """
    功能：对标签数据进行one-hot编码
    输入：1维标签数据
    输出：2维编码后数据
    备注：处理Y的标签 例如（光谱维度9,数据2） --> [ 0, 0, 1, 0, 0, 0, 0, 0, 0],此时标签没有减一处理
    """
    numClass = np.max(Y)
    y_encoded = np.zeros((Y.shape + tuple([numClass])), 'uint8')
    for i in range(1, numClass + 1):
        index = np.where(Y == i)
        if len(Y.shape) == 1:  # 如果是一维数据
            y_encoded[index[0], i - 1] = 1
        else:  # 如果是二维数据
            y_encoded[index[0], index[1], i - 1] = 1
    return y_encoded


def splitTrainTestSet(X, y, testRatio=0.90):
    """
    功能：按比例分割训练集与测试集
    输入：（原始数据X，原始标签y，验证集占比）
    输出：分割后的数据集
    备注：无
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=345, stratify=y)
    print("\n+---------------- 训练集测试集数据概览 ---------------------+")
    print('x_train.shape: ' + str(X_train.shape))
    print('y_train.shape: ' + str(y_train.shape))
    print('x_test.shape: ' + str(X_test.shape))
    print('y_test.shape: ' + str(y_test.shape))
    print("+---------------------------------------------------+")
    return X_train, X_test, y_train, y_test


def split_dataset_equal(gt_1D, numTrain=20):
    """
    功能：按照每类数量分割训练集与测试集
    输入：（一维原始标签y，每类训练数量）
    输出：训练集一维坐标,测试集一维坐标
    备注：当某类别数量过少时,就训练集测试集复用
    """
    train_idx, test_idx, numList = [], [], []
    numClass = np.max(gt_1D)  # 获取最大类别数
    for i in range(1, numClass + 1):  # 忽略背景元素
        idx = np.where(gt_1D == i)[0]  # 记录下该类别的坐标值
        numList.append(len(idx))  # 得到该类别的数量
        np.random.shuffle(idx)  # 对坐标乱序
        train_idx.append(idx[:numTrain])  # 收集每一类的训练坐标
        if len(idx) > numTrain * 2:
            test_idx.append(idx[numTrain:])  # 收集每一类的测试坐标
        else:  # 如果该类别数目过少，则训练集验证集重合使用(考虑到indianPines)
            test_idx.append(idx[-numTrain:])
    train_idx = np.asarray([item for sublist in train_idx for item in sublist])
    test_idx = np.asarray([item for sublist in test_idx for item in sublist])
    return train_idx, test_idx

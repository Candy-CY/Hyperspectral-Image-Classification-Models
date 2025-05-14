import copy
import numpy as np
import os
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def random_unison(a,b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]

def random_single(a, rstate=None):
    return a[np.random.RandomState(seed=rstate).permutation(len(a))]

def loadData(name, num_components=None, preprocessing="standard"):
    data_path = os.path.join(os.getcwd(),'../HSI-datasets')
    if name  in ["IP", "DIP", "DIPr"]:
        data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SV':
        data = sio.loadmat(os.path.join(data_path, 'salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
    elif name  in ["UP", "DUP", "DUPr"]:
        data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
    elif name == 'UH':
        data = sio.loadmat(os.path.join(data_path, 'houston.mat'))['houston']
        labels = sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_tr']
        labels += sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_te']
        num_class = 15
    else:
        print("NO DATASET")
        exit()
    num_class = 15 if name == "UH" else 9 if name in ["UP", "DUP", "DUPr"] else 16
    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])
    if num_components != None:
        data = PCA(n_components=num_components).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components
    if preprocessing == "standard": data = StandardScaler().fit_transform(data)
    elif preprocessing == "minmax": data = MinMaxScaler().fit_transform(data)
    elif preprocessing == "none": pass
    else: print("[WARNING] Not preprocessing method selected")
    data = data.reshape(shapeor)
    return data, labels, num_class


def split_data(pixels, labels, value, splitdset="sklearn", rand_state=None):
    if splitdset == "sklearn":
        X_test, X_train, y_test, y_train = \
            train_test_split(pixels, labels, test_size=value, stratify=labels, random_state=rand_state)
    elif "custom" in splitdset:
        labels = labels.reshape(-1)
        X_train = []; X_test = []; y_train = []; y_test = [];
        if "custom" == splitdset: 
            values = np.unique(value, return_counts=1)[1][1:]
            for idi, i in enumerate(values):
                samples = pixels[labels==idi+1]
                samples = random_single(samples, rstate=rand_state)
                for a in samples[:i]: 
                    X_train.append(a); y_train.append(idi)
                for a in samples[i:]:
                    X_test.append(a); y_test.append(idi)
        elif "custom2" == splitdset:
            for idi, i in enumerate(value):
                samples = pixels[labels==idi]
                samples = random_single(samples, rstate=rand_state)
                for a in samples[:i]: 
                    X_train.append(a); y_train.append(idi)
                for a in samples[i:]:
                    X_test.append(a); y_test.append(idi)
        X_train = np.array(X_train); X_test = np.array(X_test)
        y_train = np.array(y_train); y_test = np.array(y_test)
        X_train, y_train = random_unison(X_train,y_train, rstate=rand_state)
    return X_train, X_test, y_train, y_test


def select_samples(pixels, labels, samples):
    return split_data(pixels, labels, samples, splitdset="custom")

def load_split_data_fix(name, pixels, path_dset='../HSI-datasets'):
    data_path = os.path.join(os.getcwd(), path_dset)
    if name == "UH":
        y_train = sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_tr'].reshape(-1)
        y_test = sio.loadmat(os.path.join(data_path, 'houston_gt.mat'))['houston_gt_te'].reshape(-1)
    elif name in ["DIP", "DIPr"]:
        y_train2 = sio.loadmat(\
                    os.path.join(data_path, 'indianpines_disjoint_dset.mat'))\
                                            ['indianpines_disjoint_dset']
        y_test = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
        y_train = copy.deepcopy(y_train2)
        for i, val in enumerate([0,2,3,5,6,8,10,11,12,14,1,4,7,9,13,15,16]): y_train[y_train2==i] = val
        del y_train2
        if name == "DIP": y_test[y_train!=0] = 0
        else: X_train, X_test, y_train, y_test = select_samples(pixels, y_test, y_train)
    elif name in ["DUP", "DUPr"]:
        y_train = sio.loadmat(os.path.join(data_path, 'TRpavia_fixed.mat'))['TRpavia_fixed'].reshape(-1)
        y_test = sio.loadmat(os.path.join(data_path, 'TSpavia_fixed.mat'))['TSpavia_fixed'].reshape(-1)
        if name == "DUP": pass
        else: X_train, X_test, y_train, y_test = select_samples(pixels, y_test, y_train)
    if name in ["UH", "DIP", "DUP"]:
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)
        X_train = pixels[y_train!=0,:]
        X_test  = pixels[y_test!=0,:]
        del pixels
        y_train = y_train[y_train!=0] - 1
        y_test  = y_test[y_test!=0] - 1
        X_train, y_train = random_unison(X_train,y_train, rstate=None)
        #X_test, y_test = random_unison(X_test,y_test, rstate=None)
    return X_train, X_test, y_train, y_test


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
    # ALERT: TRY THIS
    #import cv2
    # return cv2.copyMakeBorder(X, margin, margin, margin, margin, cv2.BORDER_REPLICATE)


def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels.astype("int")

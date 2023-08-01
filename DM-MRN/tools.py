import torch
import torch.utils.data as Torchdata
import numpy as np
from scipy import io
import random
import os
from tqdm import tqdm
import scipy.io as sio
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from sklearn.decomposition import PCA

RESULT_DIR = './Results/'
####################################################################
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents,whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca
####################################################################
def get_dataset(dataset_name, numComponents, target_folder='./Datasets/'):
    palette = None
    folder = target_folder + dataset_name + '/'
    if dataset_name == 'PaviaU':
        img = io.loadmat(folder + 'PaviaU.mat')['paviaU']
        gt = io.loadmat(folder + 'PaviaU_gt.mat')['paviaU_gt']
        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        rgb_bands = (55, 41, 12)
        ignored_labels = [0]
    elif dataset_name == 'Salinas':
        img = io.loadmat(folder + 'Salinas_corrected.mat')['salinas_corrected']
        gt = io.loadmat(folder + 'Salinas_gt.mat')['salinas_gt']
        label_values = ['Undefined', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 
                        'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 
                        'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk', 
                        'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
        rgb_bands = (43, 21, 11)
        ignored_labels = [0]
    elif dataset_name == 'HHK':
        img = io.loadmat(folder + 'ZY_hhk.mat')['ZY_hhk_0628_data']
        gt = io.loadmat(folder + 'ZY_hhk_gt.mat')['data_gt']
        label_values = ["Undefined", "Reed", "Spartina alterniflora", "Salt filter pond",
                        "Salt evaporation pond", "Dry pond", "Tamarisk",
                        "Salt pan", "Seepweed", "River",
                        "Sea", "Mudbank", "Tidal creek",
                        "Fallow land", "Ecological restoration pond",
                        "Robinia","Fishpond","Pit pond",
                        "Building","Bare land","Paddyfield",
                        "Cotton","Soybean","Corn"]
        rgb_bands = (120, 72, 36)
        ignored_labels = [0]
    elif dataset_name == 'Houston2018':
        img = io.loadmat(folder + 'Houston2018_normalize.mat')['im']
        gt = io.loadmat(folder + 'Houston2018_gt.mat')['gt']
        label_values = ["Unclassified","Healthy grass","Stressed grass","Artificial turf",
                        "Evergreen trees","Deciduous trees","Bare earth","Water","Residential buildings",
                        "Non-residential buildings","Roads","Sidewalks","Crosswalks","Major thoroughfares",
                        "Highways","Railways","Paved parking lots","Unpaved parking lots","Cars",
                        "Trains","Stadium seats"]
        rgb_bands = (59, 40, 23)
        ignored_labels = [0]
    elif dataset_name == 'Berlin':
        img = io.loadmat(folder + 'Berlin.mat')['data_HS_LR']
        gt = io.loadmat(folder + 'Berlin_gt.mat')['gt']
        label_values = ["Undefined", "Forest","Residential Area","Industrial Area","Low Plants",
                        "Soil","Allotment","Commercial Area","Water"]
        rgb_bands = (59, 40, 23)
        ignored_labels = [0]
    else:
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)
    ignored_labels = list(set(ignored_labels))

    img = np.asarray(img, dtype='float32')
    n_bands = img.shape[-1]
    for band in range(n_bands):
        min_val = np.min(img[:,:,band])
        max_val = np.max(img[:,:,band])
        img[:,:,band] = (img[:,:,band] - min_val) / (max_val - min_val)
    if numComponents=='without':
        pass
    elif numComponents=='all':
        img, pca = applyPCA(img, numComponents=n_bands)
    else:
        img, pca = applyPCA(img, numComponents=numComponents)
    return img, gt, label_values, ignored_labels, rgb_bands, palette
####################################################################
def sample_gt(gt, train_size, mode='fixed_withone'):
    train_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
       if mode == 'random':
           train_size = float(train_size)/100
    
    if mode == 'random_withone':
        train_indices = []
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))
            train_len = int(np.ceil(train_size*len(X)))
            train_indices += random.sample(X, train_len)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0
    
    elif mode == 'fixed_withone':
        train_indices = []
        test_gt = np.copy(gt)
        index_1 = []
        special = []
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))
            if train_size > len(X) / 2:
                train_indices += random.sample(X, int(len(X)//2))
                index_1.append(c)
                special.append(len(X))
            else:
                train_indices += random.sample(X, train_size)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt, index_1, special
####################################################################
class HyperX(torch.utils.data.Dataset):
    def __init__(self, data, gt, dataset_name, patch_size=5, flip_argument=True, rotated_argument=True):
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_argument
        self.rotated_augmentation = rotated_argument
        self.name = dataset_name
        
        p = self.patch_size // 2

        if self.patch_size > 1:
            self.data = np.pad(self.data, ((p,p),(p,p),(0,0)), mode='constant')
            self.label = np.pad(self.label, p, mode='constant')
        else:
            self.flip_argument = False
            self.rotated_argument = False

        self.indices = []
        for c in np.unique(self.label):
            if c == 0:
                continue
            c_indices = np.nonzero(self.label == c)
            X = list(zip(*c_indices))
            self.indices += X
        
    def resetGt(self, gt):
        self.label = gt
        p = self.patch_size // 2
        if self.patch_size > 1:
            self.label = np.pad(gt, p, mode='constant')
            
        self.indices = []
        for c in np.unique(self.label):
            if c == 0:
                continue
            c_indices = np.nonzero(self.label == c)
            X = list(zip(*c_indices))
            self.indices += X
            
    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def rotated(*arrays):
        p = np.random.random()
        if p < 0.25:
            arrays = [np.rot90(arr) for arr in arrays]
        elif p < 0.5:
            arrays = [np.rot90(arr, 2) for arr in arrays]
        elif p < 0.75:
            arrays = [np.rot90(arr, 3) for arr in arrays]
        else:
            pass
        return arrays

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            data, label = self.flip(data, label)
        if self.rotated_augmentation and self.patch_size > 1:
            data, label = self.rotated(data, label)
        
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        data = torch.from_numpy(data)
        label = torch.from_numpy(label)


        if self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        elif self.patch_size == 1:
            label = label[0, 0]

        return data, label-1
####################################################################
def get_sample(dataset_name, sample_size, run, ITER_NUM):
    sample_file = RESULT_DIR + 'SampleSplit/' + dataset_name +'/' + str(sample_size) + '_' + str(ITER_NUM) + '/sample' + str(sample_size) + '_run' + str(run) + '.mat'
    data = io.loadmat(sample_file)
    train_gt = data['train_gt']
    test_gt = data['test_gt']
    return train_gt, test_gt
####################################################################
def save_sample(train_gt, test_gt, dataset_name, sample_size, run, SAMPLE_SIZE, ITER_NUM):
    sample_dir = RESULT_DIR + 'SampleSplit/' + dataset_name + '/' + str(SAMPLE_SIZE) + '_' + str(ITER_NUM) + '/'
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    sample_file = sample_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    io.savemat(sample_file, {'train_gt':train_gt, 'test_gt':test_gt})
####################################################################
def get_gt(dataset_name, target_folder='./Datasets/'):
    folder = target_folder + dataset_name + '/'
    if dataset_name == 'PaviaU':
        gt_PATH = folder + 'PaviaU_gt.mat'
        gt = sio.loadmat(gt_PATH)
        gt = gt['paviaU_gt']
    elif dataset_name == 'Salinas':
        gt_PATH = folder + 'Salinas_gt.mat'
        gt = sio.loadmat(gt_PATH)
        gt = gt['salinas_gt']
    elif dataset_name == 'HHK':
        gt_PATH = folder + 'ZY_hhk_gt.mat'
        gt = sio.loadmat(gt_PATH)
        gt = gt['data_gt']
    elif dataset_name == 'Houston2018':
        gt_PATH = folder + 'Houston2018_gt.mat'
        gt = sio.loadmat(gt_PATH)
        gt = gt['gt']
    elif dataset_name == 'Berlin':
        gt_PATH = folder + 'Berlin_gt.mat'
        gt = sio.loadmat(gt_PATH)
        gt = gt['gt']
    else:
        raise ValueError("{} dataset is unknown.".format(dataset_name))
    return gt
####################################################################
def save_result(result, save_folder, sample_size, run):
    scores_dir = save_folder
    scores_file = scores_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    io.savemat(scores_file,result)
    print('Results saved')
####################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.05)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    else:
        pass
####################################################################
def record_output(oa_ae, aa_ae, kappa_ae,  cla, test_time,path):
    f = open(path, 'a')
    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n'
    f.write(sentence5)
    for i in range(cla.shape[1]):
        a=[]
        for j in range(cla.shape[0]):
            a.append(cla[j][i])
        sentence6 = 'Accuarcy for class '+str(i+1)+' are: ' + str(np.mean(a)) + ' ± ' + str(np.std(a)) + '\n'
        f.write(sentence6)
    sentence7 = 'avergare testing time:' + str(np.mean(test_time)) + '\n'
    f.write(sentence7)
    sentence8 = 'testing time for each iteration are:' + str(test_time) + '\n'
    f.write(sentence8)
    f.close()
####################################################################
def matrix(dir, DATASET, SAMPLE_SIZE, ITER_NUM, N_RUNS, test_time):
    ACCU_DIR = dir
    rrresult = sio.loadmat(ACCU_DIR + 'sample' + str(SAMPLE_SIZE) + '_run0.mat')['pre_gt']
    rrresult = np.array(rrresult)
    nnn_class = rrresult.max()
    OA = []
    AA = []
    Kappa = []
    cla = np.zeros((N_RUNS, int(nnn_class)))
    gt = get_gt(DATASET)

    for i in range(N_RUNS):
        result = sio.loadmat(ACCU_DIR + 'sample' + str(SAMPLE_SIZE) + '_run' + str(i) + '.mat')['pre_gt']
        result = np.array(result)
        n_class = result.max()
        resultX = result[:,:].flatten()
        y_pred = resultX.tolist()
        gt = gt.flatten()
        y_true = gt.tolist()
        c = confusion_matrix(y_true, y_pred)
        matrix = c[1:, 1:]
        total = np.sum(matrix, axis=1)

        # OA
        right = []
        for j in range(int(n_class)):
            right.append(matrix[j, j])
        sum_right = np.sum(right)
        sum_total = np.sum(matrix)
        OA_i = sum_right / sum_total
        OA.append(OA_i)
        # AA
        accuarcy = []
        for j in range(int(n_class)):
            acc = matrix[j, j] / total[j]
            accuarcy.append(acc)
        AA_i = np.sum(accuarcy) / int(n_class)
        AA.append(AA_i)
        # Kappa
        PO = OA_i
        sum_row = np.sum(matrix, axis=1)
        sum_col = np.sum(matrix, axis=0)
        PE = 0
        for j in range(int(n_class)):
            a = sum_col[j] * sum_row[j]
            PE = PE + a
        PE = PE / (sum_total * sum_total)
        Kappa_i = (PO - PE) / (1 - PE)
        Kappa.append(Kappa_i)
        # class
        for j in range(n_class):
            cla[i][j] = right[j] / sum_row[j]

    record_output(OA, AA, Kappa, cla,test_time, ACCU_DIR + str(SAMPLE_SIZE) + '_' + str(ITER_NUM) + '.txt')
####################################################################
def drawresult(y_pred,imageid,SAMPLE_SIZE,ITER_NUM,drawlabel,dir,run):
    drawlabel = np.array(drawlabel)
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(-1,1)
    n_class = y_pred.max()
    if imageid == 'PaviaU':
        row = 610
        col = 340
        palette = np.array([[0, 0, 255],
                            [76, 230, 0],
                            [255, 190, 232],
                            [255, 0, 0],
                            [156, 156, 156],
                            [255, 255, 115],
                            [0, 255, 197],
                            [132, 0, 168],
                            [0, 0, 0]])
        palette = palette * 1.0 / 255
    elif imageid == 'Berlin':
        row = 1723
        col = 476
        palette = np.array([[0,139,0],
                            [255, 127, 80],
                            [218,112,214],
                            [0,255,0],
                            [255,255,0],
                            [255,0,255],
                            [61,185,117],
                            [0,0,255]])
        palette = palette * 1.0 / 255
    elif imageid == 'Salinas':
        row = 512
        col = 217
        palette = np.array([[0, 168, 132],
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
                            [255, 170, 0]])
        palette = palette * 1.0 / 255
    elif imageid == 'HHK':
        row = 1147
        col = 1600
        palette = np.array([[0, 139, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [255, 127, 80],
                            [255, 0, 255],
                            [139, 139, 0],
                            [0, 139, 139],
                            [0, 255, 0],
                            [0, 255, 255],
                            [0, 30, 190],
                            [127, 255, 0],
                            [218, 112, 214],
                            [46, 139, 87],
                            [0, 0, 139],
                            [255, 165, 0],
                            [127, 255, 212],
                            [218, 112, 214],
                            [255, 0, 0],
                            [205, 0, 0],
                            [139, 0, 0],
                            [65, 105, 225],
                            [240, 230, 140],
                            [244, 164, 96]])
        palette = palette * 1.0 / 255
    elif imageid == 'Houston2018':
        row = 601
        col = 2384
        palette = np.array([[176,238,0],
                    [158,214,0],
                    [117,158,0],
                    [0,139,0],
                    [0,200,0],
                    [255,255,0],
                    [0,255,255],
                    [255,175,147],
                    [255,127,80],
                    [0,133,130],
                    [167,140,167],
                    [172,86,0],
                    [0,0,255],
                    [218,112,214],
                    [235,179,232],
                    [139,139,0],
                    [213,208,0],
                    [255,2,0],
                    [205,0,0],
                    [139,0,0]])
        palette = palette * 1.0 / 255
    X_result = np.ones((y_pred.shape[0],3))
    for i in range(1,n_class+1):
        X_result[np.where(y_pred == i), 0] = palette[i - 1, 0]
        X_result[np.where(y_pred == i), 1] = palette[i - 1, 1]
        X_result[np.where(y_pred == i), 2] = palette[i - 1, 2]
    X_mask = np.zeros((row+4,col+4,3))
    X_result[np.where(drawlabel == 0), 0] = 1.0
    X_result[np.where(drawlabel == 0), 1] = 1.0
    X_result[np.where(drawlabel == 0), 2] = 1.0
    X_result = X_result.reshape(row,col,3)

    X_mask[2:-2,2:-2,:] = X_result[:,:,:]
    plt.axis("off")
    plt.imsave(dir + 'sample' + str(SAMPLE_SIZE) + '_iter' + str(ITER_NUM) + '_run' + str(run) + '.png', X_mask)
    # plt.imsave(dir + 'sample' + str(SAMPLE_SIZE) + '_iter' + str(ITER_NUM) + '_run' + str(run) + '.svg', X_mask)
    # plt.imsave(dir + 'sample' + str(SAMPLE_SIZE) + '_iter' + str(ITER_NUM) + '_run' + str(run) + '.eps', X_mask)
    return X_result
####################################################################
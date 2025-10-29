import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  
from sklearn.metrics import confusion_matrix
import torch
import os
import random
from torch import nn
from torch.nn import functional as F

class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')  # uncertainty loss weight
        return loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def featureNormalize(X,type):
    #type==1 x = (x-mean)/std(x)
    #type==2 x = (x-max(x))/(max(x)-min(x))
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
        X_norm = X_norm/(maxX-minX)
        return X_norm    
        
def DrawResult(labels,imageID):
    # ID=1:Pavia University
    # ID=2:Salinas
    # ID=3:Houston2013
    # ID=4:Indian_pines
    # ID=5:LongKou
    # ID=6:HanChuan
    # ID=7:HongHu
    # ID=8:Houston2018
    # ID=15: SZUTreeHSI-R1
    # ID=16: SZUTreeHSI-R2
    num_class = int(labels.max())
    if imageID == 1:
        row = 610
        col = 340
        palette = np.array([[216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])
        palette = palette*1.0/255
    
    elif imageID ==2:
        row = 512
        col = 217
        palette = np.array([[37, 58, 150],
                            [47, 78, 161],
                            [56, 87, 166],
                            [56, 116, 186],
                            [51, 181, 232],
                            [112, 204, 216],
                            [119, 201, 168],
                            [148, 204, 120],
                            [188, 215, 78],
                            [238, 234, 63],
                            [246, 187, 31],
                            [244, 127, 33],
                            [239, 71, 34],
                            [238, 33, 35],
                            [180, 31, 35],
                            [123, 18, 20]])
        palette = palette*1.0/255

    elif imageID == 3:
        row = 349
        col = 1905
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [0, 0, 0],
                            [255, 255, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [255, 127, 80]])
        palette = palette * 1.0 / 255

    elif imageID == 4:
        row = 145
        col = 145
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0]])
        palette = palette * 1.0 / 255

    elif imageID == 5:
        row = 550
        col = 400
        palette = np.array([[255, 0, 0],
                            [239, 155, 0],
                            [255, 255, 0],
                            [0, 255, 0],
                            [0, 255, 255],
                            [0, 140, 140],
                            [0, 0, 255],
                            [255, 255, 255],
                            [160, 32, 240]])
        palette = palette * 1.0 / 255

    elif imageID == 6:
        row = 1217
        col = 303
        palette = np.array([[176, 48, 96],
                            [0, 255, 255],
                            [255, 0, 255],
                            [160, 32, 240],
                            [127, 255, 212],
                            [127, 255, 0],
                            [0, 205, 0],
                            [0, 255, 0],
                            [0, 139, 0],
                            [255, 0, 0],
                            [216, 191, 216],
                            [255, 127, 80],
                            [160, 82, 45],
                            [255, 255, 255],
                            [218, 112, 214],
                            [0, 0, 255],
                            ])
        palette = palette * 1.0 / 255
    elif imageID == 7:
        row = 940
        col = 475
        palette = np.array([[255, 0, 0],
                            [255, 255, 255],
                            [176, 48, 96],
                            [255, 255, 0],
                            [255, 127, 80],
                            [0, 255, 0],
                            [0, 205, 0],
                            [0, 139, 0],
                            [127, 255, 212],
                            [160, 32, 240],
                            [216, 191, 216],
                            [0, 0, 255],
                            [0, 0, 139],
                            [218, 112, 214],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 165, 0],
                            [127, 255, 0],
                            [139, 139, 0],
                            [0, 139, 139],
                            [205, 181, 205],
                            [238, 154, 0]])
        palette = palette * 1.0 / 255
    elif imageID == 8:
        row = 601
        col = 2384
        palette = np.array([[0, 206, 0],
                            [123, 220, 0],
                            [47, 139, 85],
                            [0, 136, 0],
                            [0, 68, 0],
                            [159, 79, 41],
                            [68, 227, 251],
                            [255, 255, 255],
                            [213, 191, 213],
                            [248, 2, 0],
                            [167, 160, 146],
                            [124, 124, 124],
                            [160, 4, 3],
                            [80, 0, 5],
                            [226, 161, 13],
                            [255, 242, 3],
                            [237, 153, 0],
                            [242, 0, 200],
                            [0, 6, 191],
                            [172, 196, 219]
                            ])
        palette = palette * 1.0 / 255

    elif imageID == 10:
        row = 1740
        col = 860
        palette = np.array([[140,67,46],
                            [153,153,153],
                            [255,100,0],
                            [0,255,123],
                            [164,75,155],
                            [101,174,255],
                            [118,254,172],
                            [60,91,112],
                            [255,255,0],
                            [255,255,125],
                            [255,0,255],
                            [100,0,255],
                            [0,172,254],
                            [0,255,0],
                            [171,175,80],
                            [101,193,60],
                            [139,0,0],
                            [0,0,255]
                            ])
        palette = palette * 1.0 / 255

    elif imageID == 11:
        row = 880
        col = 1360
        palette = np.array([[0, 255, 0],
                            [153, 153, 153],
                            [255, 100, 0],
                            [164, 75, 155],
                            [101, 174, 255],
                            [140, 67, 46]
                            ])
        palette = palette * 1.0 / 255

    elif imageID == 12:
        row = 1230
        col = 1000
        palette = np.array([[140, 67, 46],
                            [0, 0, 255],
                            [0, 200, 0],
                            [101, 174, 255],
                            [164, 75, 155],
                            [192, 80, 70],
                            [60, 91, 112],
                            [255, 255, 0],
                            [255, 100, 0],
                            [118, 254, 172]
                            ])
        palette = palette * 1.0 / 255
    elif imageID == 15:
        row = 2405
        col = 3085
        palette = np.array([[55, 65, 62],
                            [162, 132, 163],
                            [249, 191, 0],
                            [196, 44, 198],
                            [133, 110, 211],
                            [127, 255, 255],
                            [120, 192, 207],
                            [239, 227, 217],
                            [216, 131, 221],
                            [193, 124, 148],
                            [211, 67, 111],
                            [66, 115, 196],
                            [67, 38, 120],
                            [51, 175, 71],
                            [184, 206, 194],
                            [217, 214, 219],
                            [0, 255, 1]
                            ])
        palette = palette * 1.0 / 255
    elif imageID == 16:
        row = 2444
        col = 4040
        palette = np.array([[140, 67, 46],
                            [0, 0, 255],
                            [0, 200, 0],
                            [101, 174, 255],
                            [164, 75, 155],
                            [192, 80, 70],
                            [60, 91, 112],
                            [255, 255, 0],
                            [255, 100, 0],
                            [118, 254, 172]
                            ])
        palette = palette * 1.0 / 255

    X_result = np.zeros((labels.shape[0],3))
    for i in range(1,num_class+1):
        X_result[np.where(labels==i),0] = palette[i-1,0]
        X_result[np.where(labels==i),1] = palette[i-1,1]
        X_result[np.where(labels==i),2] = palette[i-1,2]
    
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
    plt.imshow(X_result)    
    return X_result
    
def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = np.sum(predict==label)*1.0/n
    correct_sum = np.zeros((max(label)+1))
    reali = np.zeros((max(label)+1))
    predicti = np.zeros((max(label)+1))
    producerA = np.zeros((max(label)+1))
    
    for i in range(0,max(label)+1):
        correct_sum[i] = np.sum(label[np.where(predict==i)]==i)
        reali[i] = np.sum(label==i)
        predicti[i] = np.sum(predict==i)
        producerA[i] = correct_sum[i] / reali[i]
   
    Kappa = (n*np.sum(correct_sum) - np.sum(reali * predicti)) *1.0/ (n*n - np.sum(reali * predicti))
    return OA,Kappa,producerA

# def LoadHSI(dataID=1,num_label=150, val_num=10):
#     # ID=1:Pavia University
#     # ID=2:Salinas
#     # ID=3:Houston2013
#     # ID=4:Indian_pines
#     # ID=5:LongKou
#     # ID=6:HanChuan
#     # ID=7:HongHu
#     # ID=8:Houston2018
#
#     if dataID==1:
#         data = sio.loadmat('./Data/pu/PaviaU.mat')
#         X = data['paviaU']
#         data = sio.loadmat('./Data/pu/PaviaU_gt.mat')
#         Y = data['paviaU_gt']
#
#     elif dataID==2:
#         data = sio.loadmat('./Data/sa/Salinas_corrected.mat')
#         X = data['salinas_corrected']
#         data = sio.loadmat('./Data/sa/Salinas_gt.mat')
#         Y = data['salinas_gt']
#
#     elif dataID==3:
#         data = sio.loadmat('./Data/houston13/GRSS2013.mat')
#         X = data['GRSS2013']
#         data = sio.loadmat('./Data/houston13/GRSS2013_gt.mat')
#         Y = data['GRSS2013_gt']
#
#     elif dataID==4:
#         data = sio.loadmat('./Data/ip/Indian_pines_corrected.mat')
#         X = data['indian_pines_corrected']
#         data = sio.loadmat('./Data/ip/Indian_pines_gt.mat')
#         Y = data['indian_pines_gt']
#         # num_label = [15, 25, 25, 25, 25, 25, 15, 25, 15, 25, 25, 25, 25, 25, 25, 25]  # class 1、7、9
#         # num_label = [15, 50, 50, 50, 50, 50, 15, 50, 15, 50, 50, 50, 50, 50, 50, 50] #class 1、7、9
#         # num_label = [15, 75, 75, 75, 75, 75, 15, 75, 15, 75, 75, 75, 75, 75, 75, 75]  # class 1、7、9
#         # num_label = [15, 100, 100, 100, 100, 100, 15, 100, 15, 100, 100, 100, 100, 100, 100, 15]  # class 1、7、9
#         num_label = [15, 30, 30, 30, 30, 30, 15, 30, 15, 30, 30, 30, 30, 30, 30, 15]
#
#     elif dataID==5:
#         data = sio.loadmat('./Data/whulk/WHU_Hi_LongKou.mat')
#         X = data['WHU_Hi_LongKou']
#         data = sio.loadmat('./Data/whulk/WHU_Hi_LongKou_gt.mat')
#         Y = data['WHU_Hi_LongKou_gt']
#
#     elif dataID==6:
#         data = sio.loadmat('./Data/whuhc/WHU_Hi_HanChuan.mat')
#         X = data['WHU_Hi_HanChuan']
#         data = sio.loadmat('./Data/whuhc/WHU_Hi_HanChuan_gt.mat')
#         Y = data['WHU_Hi_HanChuan_gt']
#
#     elif dataID==7:
#         data = sio.loadmat('./Data/whuhh/WHU_Hi_HongHu.mat')
#         X = data['WHU_Hi_HongHu']
#         data = sio.loadmat('./Data/whuhh/WHU_Hi_HongHu_gt.mat')
#         Y = data['WHU_Hi_HongHu_gt']
#
#     elif dataID==8:
#         data = sio.loadmat('./Data/houston18/Houston2018.mat')
#         X = data['Houston2018']
#         data = sio.loadmat('./Data/houston18/Houston2018_gt.mat')
#         Y = data['Houston2018_gt']
#
#     elif dataID==10:
#         data = sio.loadmat('./Data/QUH_TDW/QUH-Tangdaowan.mat')
#         X = data['Tangdaowan']
#         data = sio.loadmat('./Data/QUH_TDW/QUH-Tangdaowan_GT.mat')
#         Y = data['TangdaowanGT']
#
#     elif dataID==11:
#         data = sio.loadmat('./Data/QUH_QY/QUH-Qingyun.mat')
#         X = data['Chengqu']
#         data = sio.loadmat('./Data/QUH_QY/QUH-Qingyun_GT.mat')
#         Y = data['ChengquGT']
#
#     elif dataID==12:
#         data = sio.loadmat('./Data/QUH_PA/QUH-Pingan.mat')
#         X = data['Haigang']
#         data = sio.loadmat('./Data/QUH_PA/QUH-Pingan_GT.mat')
#         Y = data['HaigangGT']
#
#
#
#     [row,col,n_feature] = X.shape
#     K = row*col
#     X = X.reshape(K, n_feature)
#
#     n_class = Y.max()
#
#     X = featureNormalize(X,2)
#     X = np.reshape(X,(row,col,n_feature))
#     X = np.moveaxis(X,-1,0)
#     Y = Y.reshape(K,).astype('int')
#
#
#     for i in range(1,n_class+1):
#
#         index = np.where(Y==i)[0]
#         n_data = index.shape[0]
#         np.random.seed(12345)
#         randomArray_label = np.random.permutation(n_data)
#         if dataID==4:
#             train_num = num_label[i-1]
#         else:
#             train_num = num_label
#         if i==1:
#             train_array = index[randomArray_label[0:train_num]]
#             val_array = index[randomArray_label[train_num:train_num + val_num]]
#             test_array = index[randomArray_label[train_num + val_num:n_data]]
#         else:
#             train_array = np.append(train_array,index[randomArray_label[0:train_num]])
#             val_array = np.append(val_array,index[randomArray_label[train_num:train_num + val_num]])
#             test_array = np.append(test_array,index[randomArray_label[train_num:n_data]])
#
#     return X,Y,train_array,val_array,test_array

def LoadHSI(dataID=1, num_label=150):
    # ID=1:Pavia University
    # ID=2:Salinas
    # ID=3:Houston2013
    # ID=4:Indian_pines
    # ID=5:LongKou
    # ID=6:HanChuan
    # ID=7:HongHu
    # ID=8:Houston2018

    if dataID == 1:
        data = sio.loadmat('./Data/pu/PaviaU.mat')
        X = data['paviaU']
        data = sio.loadmat('./Data/pu/PaviaU_gt.mat')
        Y = data['paviaU_gt']

    elif dataID == 2:
        data = sio.loadmat('./Data/sa/Salinas_corrected.mat')
        X = data['salinas_corrected']
        data = sio.loadmat('./Data/sa/Salinas_gt.mat')
        Y = data['salinas_gt']

    elif dataID == 3:
        data = sio.loadmat('./Data/houston13/GRSS2013.mat')
        X = data['GRSS2013']
        data = sio.loadmat('./Data/houston13/GRSS2013_gt.mat')
        Y = data['GRSS2013_gt']

    elif dataID == 4:
        data = sio.loadmat('./Data/ip/Indian_pines_corrected.mat')
        X = data['indian_pines_corrected']
        data = sio.loadmat('./Data/ip/Indian_pines_gt.mat')
        Y = data['indian_pines_gt']
        # num_label = [15, 25, 25, 25, 25, 25, 15, 25, 15, 25, 25, 25, 25, 25, 25, 25]  # class 1、7、9
        # num_label = [15, 50, 50, 50, 50, 50, 15, 50, 15, 50, 50, 50, 50, 50, 50, 50] #class 1、7、9
        # num_label = [15, 75, 75, 75, 75, 75, 15, 75, 15, 75, 75, 75, 75, 75, 75, 75]  # class 1、7、9
        # num_label = [15, 100, 100, 100, 100, 100, 15, 100, 15, 100, 100, 100, 100, 100, 100, 15]  # class 1、7、9
        # num_label = [15, 30, 30, 30, 30, 30, 15, 30, 15, 30, 30, 30, 30, 30, 30, 15]

    elif dataID == 5:
        data = sio.loadmat('./Data/whulk/WHU_Hi_LongKou.mat')
        X = data['WHU_Hi_LongKou']
        data = sio.loadmat('./Data/whulk/WHU_Hi_LongKou_gt.mat')
        Y = data['WHU_Hi_LongKou_gt']

    elif dataID == 6:
        data = sio.loadmat('./Data/whuhc/WHU_Hi_HanChuan.mat')
        X = data['WHU_Hi_HanChuan']
        data = sio.loadmat('./Data/whuhc/WHU_Hi_HanChuan_gt.mat')
        Y = data['WHU_Hi_HanChuan_gt']

    elif dataID == 7:
        data = sio.loadmat('./Data/whuhh/WHU_Hi_HongHu.mat')
        X = data['WHU_Hi_HongHu']
        data = sio.loadmat('./Data/whuhh/WHU_Hi_HongHu_gt.mat')
        Y = data['WHU_Hi_HongHu_gt']

    elif dataID == 8:
        data = sio.loadmat('./Data/houston18/Houston2018.mat')
        X = data['Houston2018']
        data = sio.loadmat('./Data/houston18/Houston2018_gt.mat')
        Y = data['Houston2018_gt']

    elif dataID == 10:
        data = sio.loadmat('./Data/QUH_TDW/QUH-Tangdaowan.mat')
        X = data['Tangdaowan']
        data = sio.loadmat('./Data/QUH_TDW/QUH-Tangdaowan_GT.mat')
        Y = data['TangdaowanGT']

    elif dataID == 11:
        data = sio.loadmat('./Data/QUH_QY/QUH-Qingyun.mat')
        X = data['Chengqu']
        data = sio.loadmat('./Data/QUH_QY/QUH-Qingyun_GT.mat')
        Y = data['ChengquGT']

    elif dataID == 12:
        data = sio.loadmat('./Data/QUH_PA/QUH-Pingan.mat')
        X = data['Haigang']
        data = sio.loadmat('./Data/QUH_PA/QUH-Pingan_GT.mat')
        Y = data['HaigangGT']

    elif dataID == 15:
        data = sio.loadmat('./Data/SZUTreeHSI-R1/tree1.mat')
        X = data['data']
        data = sio.loadmat('./Data/SZUTreeHSI-R1/tree1_gt.mat')
        Y = data['gt']

    [row, col, n_feature] = X.shape
    K = row * col
    X = X.reshape(K, n_feature)

    n_class = Y.max()

    X = featureNormalize(X, 2)
    X = np.reshape(X, (row, col, n_feature))
    X = np.moveaxis(X, -1, 0)
    Y = Y.reshape(K, ).astype('int')

    for i in range(1, n_class + 1):

        index = np.where(Y == i)[0]
        n_data = index.shape[0]
        np.random.seed(12345)
        randomArray_label = np.random.permutation(n_data)
        train_num = num_label
        # train_num = num_label[i-1]
        if i == 1:
            train_array = index[randomArray_label[0:train_num]]
            test_array = index[randomArray_label[train_num:n_data]]
        else:
            train_array = np.append(train_array, index[randomArray_label[0:train_num]])
            test_array = np.append(test_array, index[randomArray_label[train_num:n_data]])

    return X, Y, train_array, test_array


def PCANorm(X, num_PC):
    mu = np.mean(X, 0)
    X_norm = X - mu

    Sigma = np.cov(X_norm.T)
    [U, _, _] = np.linalg.svd(Sigma)
    XPCANorm = np.dot(X_norm, U[:, 0:num_PC])
    return XPCANorm

def Get_train_and_test_data(img_size, img):
    H0, W0, C = img.shape
    if H0<img_size:
        gap = img_size-H0
        mirror_img = img[(H0-gap):H0,:,:]
        img = np.concatenate([img,mirror_img],axis=0)
    if W0<img_size:
        gap = img_size-W0
        mirror_img = img[:,(W0 - gap):W0,:]
        img = np.concatenate([img,mirror_img],axis=1)
    H, W, C = img.shape

    num_H = H // img_size
    num_W = W // img_size
    sub_H = H % img_size
    sub_W = W % img_size
    if sub_H != 0:
        gap = (num_H+1)*img_size - H
        mirror_img = img[(H - gap):H, :, :]
        img = np.concatenate([img, mirror_img], axis=0)

    if sub_W != 0:
        gap = (num_W + 1) * img_size - W
        mirror_img = img[:, (W - gap):W, :]
        img = np.concatenate([img, mirror_img], axis=1)
        # gap = img_size - num_W*img_size
        # img = img[:,(W - gap):W,:]
    H, W, C = img.shape
    print('padding img:', img.shape)

    num_H = H // img_size
    num_W = W // img_size

    sub_imgs = []
    for i in range(num_H):
        for j in range(num_W):
            z = img[i * img_size:(i + 1) * img_size, j * img_size:(j + 1) * img_size, :]
            sub_imgs.append(z)
    sub_imgs = np.array(sub_imgs)  # [num_H*num_W,img_size,img_size, C ]
    return sub_imgs, num_H, num_W

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()

def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        x , batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def get_Traindata_(img_size,img, data_label, train_number, batch_size):
    seed=1
    np.random.seed(seed)
    random.seed(seed)


    uc_position = np.array(np.where(data_label == 1)).transpose(1, 0)
    c_position = np.array(np.where(data_label == 2)).transpose(1, 0)
    selected_uc = np.random.choice(uc_position.shape[0], int(train_number), replace=False)
    selected_c = np.random.choice(c_position.shape[0], int(train_number), replace=False)

    selected_uc_position = uc_position[selected_uc]  # (500,),adarray
    selected_c_position = c_position[selected_c]

    TR = np.zeros(data_label.shape)
    for i in range(int(train_number)):
        TR[selected_c_position[i][0], selected_c_position[i][1]] = 1
        TR[selected_uc_position[i][0], selected_uc_position[i][1]] = 2
    # --------------测试样本-----------------
    # TE=data_label-TR
    TE = data_label  # all the data are inputt for test
    num_classes = np.max(TR)
    num_classes = int(num_classes)

    height, width, band = img.shape
    print("height={0},width={1},band={2}".format(height, width, band))

    total_pos_train, total_pos_test, number_train, number_test = chooose_train_and_test_point(TR, TE, num_classes) # for GF5B


    data_label0 = np.zeros(data_label.shape) - 1
    for i in range(int(train_number)):
        data_label0[selected_c_position[i][0], selected_c_position[i][1]] = 1
        data_label0[selected_uc_position[i][0], selected_uc_position[i][1]] = 0
    data_label0 = data_label0[:, :, np.newaxis]
    mirror_img = mirror_hsi(height, width, band, img, patch=img_size)
    mirror_image_label = mirror_hsi(height, width, 1, data_label0, patch=img_size)

    total_pos_test = total_pos_test[:2, :]
    img_train, _ = get_train_and_test_data(mirror_img, band, total_pos_train, total_pos_test,
                                                         patch=img_size, band_patch=1)
    label_train, _ = get_train_and_test_data(mirror_image_label, 1, total_pos_train, total_pos_test,
                                                          patch=img_size, band_patch=1)
    # -------------------------------------------------------------------------------
    # load data
    img_train = torch.from_numpy(img_train.transpose(0,3,1,2)).type(torch.FloatTensor)  # [1000, 155, 5, 5]
    label_train = torch.from_numpy(label_train).type(torch.LongTensor)  # [695]
    Label_train = Data.TensorDataset(img_train, label_train)
    label_train_loader = Data.DataLoader(Label_train, batch_size=batch_size, shuffle=True)

    return label_train_loader


def chooose_train_and_test_point(train_data, test_data, num_classes):
    H,W=train_data.shape
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    # number_true = []
    # pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    total_pos_test = total_pos_test.astype(int)
    #
    # # for local labeled GF5B dataset
    # if dataset=='GF5B_BI':
    #     uncertain = np.argwhere(test_data == 0)
    #     number_test.append(uncertain.shape[0])
    #     total_pos_test = np.r_[pos_test[0], pos_test[1], uncertain]
    #     total_pos_test = total_pos_test.astype(int)
    # else:
    #     total_pos_test = pos_test[0]
    #     for i in range(1, num_classes):
    #         total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    #     total_pos_test = total_pos_test.astype(int)

    return total_pos_train, total_pos_test, number_train, number_test
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(input_normalize[:,:,50])
    # plt.subplot(1, 2, 2)
    # plt.imshow(mirror_hsi[:, :, 50])


    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image
def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
def get_train_and_test_data(mirror_image, band, train_point, test_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    # x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("**************************************************")
    return x_train.squeeze(), x_test.squeeze()

def compute_loss2(self, batch_pred,batch_target,  criterion):
    b, h, w = batch_target.shape
    # batch_target = batch_target.reshape(1, -1)
    batch_target = batch_target.reshape(b, -1)

    # idx_c = (batch_target == 1).nonzero()  # change sample
    # idx_u = (batch_target == 2).nonzero()  # unchange sample

    idx_c = torch.where(batch_target == 2)  # change sample:2
    idx_u = torch.where(batch_target == 1)  # unchange sample:1
    # batch_pred;[Batch_size, 2, H,W]
    batch_pred = batch_pred.permute(1, 0, 2, 3)  # [2,Batch_size, H,W]
    batch_pred = torch.reshape(batch_pred, [2, b, -1])  # [2,Batch_size,-1]
    batch_pred_c = batch_pred[:, idx_c[0], idx_c[1]]
    batch_pred_u = batch_pred[:, idx_u[0], idx_u[1]]
    batch_pred = torch.cat([batch_pred_c, batch_pred_u], dim=1).squeeze()  # [Batch_size,-1]
    batch_pred = batch_pred.permute(1, 0)  # [Batch_size,-1, 2]

    # batch_pred = batch_pred.permute(1,2,0) # [Batch_size,-1, 2]
    # batch_pred = batch_pred.reshape([-1, 2])  # [N, 2]

    idx_c = batch_target[idx_c]
    # idx_c = batch_target[idx_c[0], idx_c[1]]
    idx_u = batch_target[idx_u]
    batch_target = torch.cat([idx_c, idx_u], dim=0).squeeze()  # [Batch_size,-1]
    # batch_target=batch_target.reshape([-1,1]).squeeze()

    loss = criterion(batch_pred, batch_target)
    return loss
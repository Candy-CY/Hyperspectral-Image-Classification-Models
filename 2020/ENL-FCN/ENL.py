import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.utils.data as pydata
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
from sklearn import preprocessing
import time
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torch.autograd as autograd
import torch.cuda.comm as comm
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load
import os
import functools
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

class HData(Dataset):
    def __init__(self, dataset, tranform=None):
        self.data = dataset[0]
        self.trans=tranform
        self.labels = dataset[1]
    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index,:,:,:]))
        label=torch.from_numpy(np.asarray(self.labels[index,:,:]))
        return img, label
    def __len__(self):
        return len(self.labels)
    def __labels__(self):
        return self.labels

def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")

class CA_Weight(autograd.Function):
    @staticmethod
    def forward(ctx, t, f):
        # Save context
        n, c, h, w = t.size()
        size = (n, h+w-1, h, w)
        weight = torch.zeros(size, dtype=t.dtype, layout=t.layout, device=t.device)
        _ext.ca_forward_cuda(t, f, weight)
        # Output
        ctx.save_for_backward(t, f)
        return weight
    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors
        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)
        _ext.ca_backward_cuda(dw.contiguous(), t, f, dt, df)
        _check_contiguous(dt, df)
        return dt, df

class CA_Map(autograd.Function):
    @staticmethod
    def forward(ctx, weight, g):
        # Save context
        out = torch.zeros_like(g)
        _ext.ca_map_forward_cuda(weight, g, out)
        # Output
        ctx.save_for_backward(weight, g)
        return out
    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors
        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)
        _ext.ca_map_backward_cuda(dout.contiguous(), weight, g, dw, dg)
        _check_contiguous(dw, dg)
        return dw, dg
ca_weight = CA_Weight.apply
ca_map = CA_Map.apply

class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self,in_dim):
        super(CrissCrossAttention,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = 0.6#nn.Parameter(torch.zeros(1))
    def forward(self,x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)
        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma*out + x
        return out
__all__ = ["CrissCrossAttention", "ca_weight", "ca_map"]

class SSCDNonLModel(nn.Module):
    def __init__(self, num_classes, n_bands, chanel):
        super(SSCDNonLModel, self).__init__()
        #self.num_Node=num_Node
        self.bands=n_bands
        chanel=chanel
        kernel=5
        CCChannel=25
        self.b1=nn.BatchNorm2d(self.bands)
        self.con1=nn.Conv2d(self.bands, chanel, 1, padding=0,bias=True)
        self.s1=nn.Sigmoid()
        self.cond1=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd1=nn.Sigmoid()

        self.b2=nn.BatchNorm2d(self.bands+chanel)
        self.con2=nn.Conv2d(self.bands+chanel, chanel, 1, padding=0,bias=True)
        self.s2=nn.Sigmoid()
        self.cond2=nn.Conv2d(chanel, CCChannel, kernel, padding=2, groups=25, bias=True)
        self.sd2=nn.Sigmoid()

        self.b4=nn.BatchNorm2d(CCChannel)
        self.nlcon2=CrissCrossAttention(CCChannel)
        self.nlcon3=CrissCrossAttention(CCChannel)
        self.bcat=nn.BatchNorm2d(CCChannel+CCChannel)
        self.con4=nn.Conv2d(CCChannel+CCChannel, chanel, 1, padding=0, bias=True)
        self.s4=nn.Sigmoid()
        self.cond4=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd4=nn.Sigmoid()

        self.b5=nn.BatchNorm2d(CCChannel+chanel)
        self.con5=nn.Conv2d(CCChannel+chanel, chanel, 1, padding=0, bias=True)
        self.s5=nn.Sigmoid()
        self.cond5=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd5=nn.Sigmoid()

        self.con6=nn.Conv2d(chanel+CCChannel, num_classes+1, 1, padding=0, bias=True)

    def forward(self, x):
        n = x.size(0)
        H=x.size(2)
        W=x.size(3)
        out1=self.b1(x)
        out1=self.con1(out1)
        out1=self.s1(out1)
        out1=self.cond1(out1)
        out1=self.sd1(out1)

        out2=torch.cat((out1,x),1)
        out2=self.b2(out2)
        out2=self.con2(out2)
        out2=self.s2(out2)
        out2=self.cond2(out2)
        out2=self.sd2(out2)

        xx=self.b4(out2)
        nl2=self.nlcon2(xx)
        nl2=self.nlcon2(nl2)
        nl3=self.nlcon3(xx)
        nl3=self.nlcon3(nl3)
        nl2=(nl2+nl3)*0.7+xx

        out4=torch.cat((xx, nl2),1)
        out4=self.bcat(out4)
        out4=self.con4(out4)
        out4=self.s4(out4)
        out4=self.cond4(out4)
        out4=self.sd4(out4)

        out5=torch.cat((out4,out2),1)
        out5=self.b5(out5)
        out5=self.con5(out5)
        out5=self.s5(out5)
        out5=self.cond5(out5)
        out5=self.sd5(out5)

        out6=torch.cat((out5,out2),1)
        out6=self.con6(out6)
        return out6

def accuracy(output, target, classcount):
    # Computes the precision k for the specified values of k
    output=output.view(classcount,-1)
    target=target.view(1,-1)
    m,n=output.size()
    _,L_output=torch.topk(output, 1, 0, True)
    count=0
    aa=0
    for i in range(n):
        if target[0,i]!=0 and L_output[0,i]==target[0,i]:
            aa=aa+1
        if target[0,i]!=0:
            count=count+1
    return aa, count

def ClassificationAccuracy(output, target, classcount):
    # Computes the precision@k for the specified values of k
    m, n = output.shape
    correct_perclass=np.zeros([classcount-1])
    count_perclass = np.zeros([classcount-1])
    count=0
    aa=0
    for i in range(m):
        for j in range(n):
            if target[i, j]!=0:
                count=count+1
                count_perclass[int(target[i,j]-1)] += 1
                if output[i, j]==target[i, j]:
                    aa=aa+1
                    correct_perclass[int(target[i,j]-1)] += 1
    test_AC_list = correct_perclass / count_perclass
    test_AA = np.average(test_AC_list)
    test_OA=aa/count
    return test_AC_list, test_OA, test_AA, aa, count

def Kappa(output, target, classcount):
    # Computes the precision@k for the specified values of k
    output=output
    target=target
    sizeOutput=np.shape(output)
    m=sizeOutput[0]
    n=sizeOutput[1]
    test_pre_label_list = []
    test_real_label_list = []
    for ii in range(m):
        for jj in range(n):
            if target[ii][jj] != 0:
                test_pre_label_list.append(output[ii][jj])
                test_real_label_list.append(target[ii][jj])
    test_pre_label_list = np.array(test_pre_label_list)
    test_real_label_list = np.array(test_real_label_list)
    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16), test_real_label_list.astype(np.int16))
    return kappa

def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass

def SpiltHSI(data, gt, split_size, edge):
    '''
    split HSI data with given slice_number
    :param data: 3D HSI data
    :param gt: 2D ground truth
    :param split_size: [height_slice,width_slice]
    :return: splited data and corresponding gt
    '''
    e = edge  # 补边像素个数
    split_height = split_size[0]
    split_width = split_size[1]
    m, n, d = data.shape
    GT=gt
    # 将无法整除的块补0变为可整除
    if m % split_height != 0 or n % split_width != 0:
        data = np.pad(data, [[0, split_height - m % split_height], [0, split_width - n % split_width], [0, 0]],
                      mode='constant')
        GT = np.pad(GT, [[0, split_height - m % split_height], [0, split_width - n % split_width]],
                    mode='constant')
    m_height = int(data.shape[0] / split_height)
    m_width = int(data.shape[1] / split_width)
    pad_data = np.pad(data, [[e, e], [e, e], [0, 0]], mode="constant")
    pad_GT = np.pad(GT, [[e, e], [e, e]], mode="constant")
    final_data = []
    final_gt=[]
    for i in range(split_height):
        for j in range(split_width):
            temp1 = pad_data[i * m_height:i * m_height + m_height + 2 * e, j * m_width:j * m_width + m_width + 2 * e, :]
            temp2 = pad_GT[i * m_height:i * m_height + m_height + 2 * e, j * m_width:j * m_width + m_width + 2 * e]
            final_data.append(temp1)
            final_gt.append(temp2)
    final_data = np.array(final_data)
    final_gt = np.array(final_gt)
    return final_data, final_gt

def PatchStack(OutPut, m, n, patch_height, patch_width, split_height, split_width, EDGE, class_count):
    HSI_stack = np.zeros([split_height * patch_height, split_width * patch_width, class_count], dtype=np.float32)
    for i in range(split_height):
        for j in range(split_width):
            if EDGE == 0:
                HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = OutPut[
                                                                                                                   i * split_width + j][
                                                                                                               EDGE:,
                                                                                                               EDGE:,
                                                                                                               :]
            else:
                HSI_stack[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = OutPut[
                                                                                                                   i * split_width + j][
                                                                                                               EDGE:-EDGE,
                                                                                                               EDGE:-EDGE, :]
    HSI_stack = np.argmax(HSI_stack, axis=2)
    HSI_stack = HSI_stack[0: -(split_height - m % split_height), 0: -(split_width - n % split_width)]
    return HSI_stack

samples_type=['ratio','same_num'][0]

#in parallel
for (FLAG,curr_train_ratio) in [(1, 0.1)]:
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    # Seed_List=[0,1,2,3,4]#随机种子点
    Seed_List=[0]#随机种子点
    if FLAG == 1:
        data_mat = sio.loadmat('../input/hybridsn/data/Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('../input/hybridsn/data/Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']
        # 参数预设
        train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01 # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 16  # 样本类别数
        learning_rate = 5e-4  # 学习率
        weight_decay = 2e-5
        max_epoch = 100#1800  # 迭代次数
        split_height = 1
        split_width = 1
        dataset_name = "indian"  # 数据集名称
        pass
    if FLAG == 2:
        data_mat = sio.loadmat('../input/hybridsn/data/PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('../input/hybridsn/data/PaviaU_gt.mat')
        gt = gt_mat['paviaU_gt']
        # 参数预设
        train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 9  # 样本类别数
        learning_rate = 5e-4  # 学习率
        max_epoch = 1500  # 迭代次数
        weight_decay = 2e-5
        split_height = 3
        split_width = 4
        EDGE=5
        dataset_name = "paviaU"  # 数据集名称
        pass
    '''
    if FLAG == 5:
        data_mat = sio.loadmat('./Datasets/KSC/KSC.mat')
        data = data_mat['KSC']
        gt_mat = sio.loadmat('./Datasets/KSC/KSC_gt.mat')
        gt = gt_mat['KSC_gt']
        # 参数预设
        # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 13  # 样本类别数
        learning_rate = 5e-4  # 学习率
        weight_decay = 2e-5
        learning_rate_sigma = 0.005
        max_epoch = 1100  # 迭代次数
        dataset_name = "KSC_BF"  # 数据集名称
        split_height = 4
        split_width = 4
        First_Chanels = 128
        After_Chanels = 32
        SIGMA = 1
        pass
    '''
    #当定义为每类样本个数时,则该参数更改为训练样本数
    train_samples_per_class=curr_train_ratio
    val_samples=class_count
    train_ratio=curr_train_ratio
    if split_height == split_width == 1:
            EDGE = 0
    else:
            EDGE = 5
    cmap = cm.get_cmap('jet', class_count + 1)
    plt.set_cmap(cmap)
    m, n, d = data.shape  # 高光谱数据的三个维度
    n_bands=d
    data = np.reshape(data, [m * n, d])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [m, n, d])
    for curr_seed in Seed_List:
        # step2:随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
        random.seed(curr_seed)
        gt_reshape = np.reshape(gt, [-1])
        train_rand_idx = []
        val_rand_idx = []
        if samples_type=='ratio':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                rand_idx = random.sample(rand_list, np.ceil(samplesCount * train_ratio).astype('int32'))
                # 随机数数量 四舍五入(改为上取整)
                rand_real_idx_per_class = idx[rand_idx]
                train_rand_idx.append(rand_real_idx_per_class)
            train_rand_idx = np.array(train_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)
            ##将测试集（所有样本，包括训练样本）也转化为特定形式
            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)
            # 背景像元的标签
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx
            # 从测试集中随机选取部分样本作为验证集
            val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集
            # 将训练集 验证集 测试集 整理
            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)

        if samples_type=='same_num':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                real_train_samples_per_class=train_samples_per_class
                rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                if real_train_samples_per_class>=samplesCount:
                    real_train_samples_per_class=int(train_samples_per_class/2)
                rand_idx = random.sample(rand_list,
                                         real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
                rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                train_rand_idx.append(rand_real_idx_per_class_train)
            train_rand_idx = np.array(train_rand_idx)
            val_rand_idx = np.array(val_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)
            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)
            # 背景像元的标签
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index  - background_idx
            # 从测试集中随机选取部分样本作为验证集
            val_data_count = int(val_samples)  # 验证集数量
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            test_data_index=test_data_index-val_data_index
            # 将训练集 验证集 测试集 整理
            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)
        # 获取训练样本的标签图
        train_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(train_data_index)):
            train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
            pass
        Train_Label=np.reshape(train_samples_gt, [m,n])
        # 获取测试样本的标签图
        test_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(test_data_index)):
            test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
            pass
        Test_Label = np.reshape(test_samples_gt, [m, n])  # 测试样本图
        # 获取验证集样本的标签图
        val_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(val_data_index)):
            val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
            pass
        Val_Label=np.reshape(val_samples_gt,[m,n])
        #############将train 和 test 和val 样本标签转化为向量形式###################
        # 训练集
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        train_samples_gt_vector = np.zeros([m * n, class_count], np.float)
        for i in range(train_samples_gt.shape[0]):
            class_idx = train_samples_gt[i]
            if class_idx != 0:
                temp = np.zeros([class_count])
                temp[int(class_idx - 1)] = 1
                train_samples_gt_vector[i] = temp
        train_samples_gt_vector = np.reshape(train_samples_gt_vector, [m, n, class_count])
        # 测试集
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        test_samples_gt_vector = np.zeros([m * n, class_count], np.float)
        for i in range(test_samples_gt.shape[0]):
            class_idx = test_samples_gt[i]
            if class_idx != 0:
                temp = np.zeros([class_count])
                temp[int(class_idx - 1)] = 1
                test_samples_gt_vector[i] = temp
        test_samples_gt_vector = np.reshape(test_samples_gt_vector, [m, n, class_count])
        # 验证集
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        val_samples_gt_vector = np.zeros([m * n, class_count], np.float)
        for i in range(val_samples_gt.shape[0]):
            class_idx = val_samples_gt[i]
            if class_idx != 0:
                temp = np.zeros([class_count])
                temp[int(class_idx - 1)] = 1
                val_samples_gt_vector[i] = temp
        val_samples_gt_vector = np.reshape(val_samples_gt_vector, [m, n, class_count])
        ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
        # 训练集
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m, n, class_count])
        # 测试集
        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m, n, class_count])
        # 验证集
        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m, n, class_count])
        Train_Split_Data, Train_Split_GT = SpiltHSI(data, Train_Label, [split_height, split_width], EDGE)
        Test_Split_Data, Test_Split_GT = SpiltHSI(data, Test_Label, [split_height, split_width], EDGE)
        _, patch_height, patch_width, bands = Train_Split_Data.shape
        patch_height -= EDGE * 2
        patch_width -= EDGE * 2
        zero_vector = np.zeros([class_count])
        all_label_mask = np.ones([1, m, n, class_count])  # 设置一个全1的mask，使得网络输出所有分类标签
    train_h=HData((np.transpose(Train_Split_Data,(0,3,1,2)).astype("float32"), Train_Split_GT), None)
    test_h=HData((np.transpose(Test_Split_Data,(0,3,1,2)).astype("float32"), Test_Split_GT), None)
    trainloader=torch.utils.data.DataLoader(train_h)
    testloader=torch.utils.data.DataLoader(test_h)
    use_cuda = torch.cuda.is_available()
    model = SSCDNonLModel(class_count, n_bands, 150) # Criss Cross Model CCNet 2B parallel
    print(model)
    if use_cuda: torch.backends.cudnn.benchmark = True
    if use_cuda: model.cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-5)
    print('lr: ',learning_rate, '  weight_dacay: ', weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-4, nesterov=True)
    best_acc = -1
    for eep in range(max_epoch):
        for batch_idx, (inputs, labels) in enumerate(trainloader):#batch_idx是enumerate（）函数自带的索引，从0开始
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
            optimizer.zero_grad()
            output= model(inputs)
            loss=criterion(output, labels.long())
            optimizer.zero_grad()   # 所有参数的梯度清零
            loss.backward()         #即反向传播求梯度
            optimizer.step()        #调用optimizer进行梯度下降更新参数
        if eep%10==0:
            Output=[]
            for Testbatch_idx, (Testinputs, Testtargets) in enumerate(testloader):#batch_idx是enumerate（）函数自带的索引，从0开始
                if use_cuda:
                    Testinputs, Testtargets = Testinputs.cuda(), Testtargets.cuda()
                Testinputs, Testtargets = torch.autograd.Variable(Testinputs), torch.autograd.Variable(Testtargets)
                Testoutput = model(Testinputs)
                Testoutput=Testoutput.data.cpu().numpy()
                Testoutput = np.transpose(Testoutput,(0,2,3,1))
                Output.append(Testoutput[0])
            OutputWhole = PatchStack(Output, m, n, patch_height, patch_width, split_height, split_width, EDGE, class_count+1)
            AC, OA, AA, rightNum, testNum= ClassificationAccuracy(OutputWhole, Test_Label, class_count+1)
            kappa = Kappa(OutputWhole, Test_Label, class_count+1)
            print("eep", eep, " test", rightNum, testNum, "OA", OA, "AA", AA, "kappa", kappa)
            print(OA, AA, kappa, AC)
        if eep==260 :
            OA=np.round(OA*100, decimals=2)
            OutputWhole = PatchStack(Output, m, n, patch_height, patch_width, split_height, split_width, EDGE, class_count+1)
            Draw_Classification_Map(OutputWhole, 'ResultsImage/' + dataset_name + '_CC2B_' + str(train_ratio) + '_' + str(OA))
        if loss.data<=0.00005:
            break
    model.train()
    model.eval()

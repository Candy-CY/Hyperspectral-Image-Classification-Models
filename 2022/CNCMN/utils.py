import numpy as np
import matplotlib.pyplot as plt
import spectral as spy
from torch.utils.data import Dataset#, DataLoader
import torch
import random
import cv2
import torch.utils.data

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

def GT_To_One_Hot(gt, class_count):
    '''
    Convet Gt to one-hot labels
    :param gt:
    :param class_count:
    :return:
    '''
    h,w=gt.shape
    GT_One_Hot = []  # 转化为one-hot形式的标签
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count,dtype=np.float32)
            if gt[i, j] != 0:
                temp[int( gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [h, w, class_count])
    return GT_One_Hot


class PatchDataset(Dataset):#需要继承data.Dataset
    def __init__(self,datalist,graphlist, is_Already_to_gpu=False, device="cpu"):
        # TODO
        self.datalist=datalist
        self.graphlist=graphlist
        self.device=device
        self.on_gpu=is_Already_to_gpu

    def __getitem__(self, index):
        # TODO
        if self.on_gpu==True:
            # do
            sample=self.datalist[index][0]
            label=self.datalist[index][1]
            subgraph=self.graphlist[index]
        else:
            # sample=torch.from_numpy(self.datalist[index][0]).to(self.device)
            sample=self.datalist[index][0].to(self.device)
            # label=torch.from_numpy(self.datalist[index][1]).to(self.device)
            label=self.datalist[index][1].to(self.device)
            # subgraph=torch.from_numpy(self.graphlist[index][0]).to(self.device)
            # subedges=torch.from_numpy(self.graphlist[index][1]).to(self.device)
            subgraph=self.graphlist[index].to(self.device)

        # return sample,label,subgraph,subedges
        return sample,label,subgraph
        #这里需要注意的是，第一步：read one data，是一个data
        pass
    def __len__(self):
        # 您应该将0更改为数据集的总大小。
        return len(self.datalist)
    


class GetDataset(object):
    def __init__(self,HSI,extend=2):
        HSI=np.array(HSI,np.float32)
        self.extend=extend
        self.H,self.W,self.B=HSI.shape
        self.HSI = np.pad(HSI, ((extend, extend), (extend, extend), (0, 0)), mode='constant')

    def getSamples(self, gt: np.array, rotation:int=0, convert2tensor:bool=True):
        if len(gt.shape) == 3:
            h, w, c = gt.shape
            gtFlag = np.sum(gt, axis=-1, keepdims=False)
        else:
            h, w = gt.shape
            gtFlag = gt
        # 对每一个非0标签样本进行切分
        patch_len=self.extend*2+1
        samples = []
        for i in range(h):
            for j in range(w):
                if gtFlag[i, j].any() == 0: continue
                # 由于HSI扩充,(i,j)像素在HSI中的实际坐标为(i+self.extend,j+self.extend)
                datacube=self.HSI[i:i+patch_len,j:j+patch_len,:]
                if rotation!=0:
                    M = cv2.getRotationMatrix2D((self.extend, self.extend), rotation, 1)
                    # 第三个参数：变换后的图像大小
                    datacube = cv2.warpAffine(datacube, M, (patch_len, patch_len))
                
                sample = []
                if convert2tensor:
                    sample.append(torch.from_numpy(datacube))
                    sample.append(torch.from_numpy(gt[i, j]))
                else:
                    sample.append(datacube)
                    sample.append(gt[i, j])

                samples.append(sample)

                # plt.show()
        return samples


def get_Samples_GT(seed: int, gt: np.array, class_count: int, train_ratio, val_ratio, samples_type: str = 'ratio', ):
    # step2:随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
    random.seed(seed)
    [height, width] = gt.shape
    gt_reshape = np.reshape(gt, [-1])
    train_rand_idx = []
    val_rand_idx = []
    if samples_type == 'ratio':
        train_number_per_class = []
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            rand_idx = random.sample(rand_list,
                                     np.ceil(samplesCount * train_ratio).astype('int32') + \
                                     np.ceil(samplesCount * val_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
            train_number_per_class.append(np.ceil(samplesCount * train_ratio).astype('int32'))
            rand_real_idx_per_class = idx[rand_idx]
            train_rand_idx.append(rand_real_idx_per_class)
        train_rand_idx = np.array(train_rand_idx)
        train_data_index = []
        val_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = list(train_rand_idx[c])
            train_data_index = train_data_index + a[:train_number_per_class[c]]
            val_data_index = val_data_index + a[train_number_per_class[c]:]
            # for j in range(a.shape[0]):
            #     train_data_index.append(a[j][0:train_number_per_class])
            #     val_data_index.append()
        # train_data_index = np.array(train_data_index).reshape([-1])
        # val_data_index = np.array(val_data_index).reshape([-1])
        
        ##将测试集（所有样本，包括训练样本）也转化为特定形式
        train_data_index = set(train_data_index)
        val_data_index = set(val_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)
        
        # 背景像元的标签
        # background_idx = np.where(gt_reshape == 0)[-1]
        # background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - val_data_index
        
        # # 从测试集中随机选取部分样本作为验证集
        # val_data_count = np.ceil(val_ratio * (len(test_data_index) + len(train_data_index))).astype('int32')  # 验证集数量
        # val_data_index = random.sample(test_data_index, val_data_count)
        # val_data_index = set(val_data_index)
        
        # test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集
        
        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)
    
    if samples_type == 'same_num':
        if int(train_ratio) == 0 or int(val_ratio) == 0:
            print("ERROR: The number of samples for train. or val. is equal to 0.")
            exit(-1)
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            real_train_samples_per_class = int(train_ratio)  # 每类相同数量样本,则训练比例为每类样本数量
            real_val_samples_per_class = int(val_ratio)  # 每类相同数量样本,则训练比例为每类样本数量
            
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            if real_train_samples_per_class >= samplesCount:
                real_train_samples_per_class = samplesCount - 1
                real_val_samples_per_class = 1
            else:
                real_val_samples_per_class = real_val_samples_per_class if (
                                                                                       real_val_samples_per_class + real_train_samples_per_class) <= samplesCount else samplesCount - real_train_samples_per_class
            rand_idx = random.sample(rand_list,
                                     real_train_samples_per_class + real_val_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)
            if real_val_samples_per_class > 0:
                rand_real_idx_per_class_val = idx[rand_idx[-real_val_samples_per_class:]]
                val_rand_idx.append(rand_real_idx_per_class_val)
        
        train_rand_idx = np.array(train_rand_idx)
        val_rand_idx = np.array(val_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)
        
        val_data_index = []
        for c in range(val_rand_idx.shape[0]):
            a = val_rand_idx[c]
            for j in range(a.shape[0]):
                val_data_index.append(a[j])
        val_data_index = np.array(val_data_index)
        
        train_data_index = set(train_data_index)
        val_data_index = set(val_data_index)
        
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)
        
        # 背景像元的标签
        # background_idx = np.where(gt_reshape == 0)[-1]
        # background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - val_data_index
        
        # # 从测试集中随机选取部分样本作为验证集
        # val_data_count = int(val_samples)  # 验证集数量
        # val_data_index = random.sample(test_data_index, val_data_count)
        # val_data_index = set(val_data_index)
        
        # test_data_index = test_data_index - val_data_index
        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)
    
    # 获取训练样本的标签图
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass
    
    # 获取测试样本的标签图
    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass
    
    # 获取验证集样本的标签图
    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        pass
    
    train_samples_gt = np.reshape(train_samples_gt, [height, width])
    test_samples_gt = np.reshape(test_samples_gt, [height, width])
    val_samples_gt = np.reshape(val_samples_gt, [height, width])
    
    return train_samples_gt, test_samples_gt, val_samples_gt


def LabelProcess(labels):
    '''
    对labels做后处理，防止出现label不连续现象
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))
    
    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i
    
    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels

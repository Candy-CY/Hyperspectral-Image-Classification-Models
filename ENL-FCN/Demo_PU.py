# pytorch 版本 SSCDenseNet

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.utils.data as pydata
from torchsummary import summary
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
from H_datapy import *

import CCnet2BModel1 as CC2B
import CCnetConcat2B1 as CCCon2B

import torch.nn.functional as F
from autis import *
#from functions import CrissCrossAttention

samples_type=['ratio','same_num'][0]


for (FLAG,curr_train_ratio) in [(2, 0.01)]:
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []

    # Seed_List=[0,1,2,3,4]#随机种子点
    Seed_List=[0]#随机种子点

    if FLAG == 1:
        data_mat = sio.loadmat('./Datasets/IndianPines/Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('./Datasets/IndianPines/Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']
        # 参数预设
        train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0 # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 16  # 样本类别数
        learning_rate = 5e-4  # 学习率
        weight_decay = 2e-5
        max_epoch = 1000  # 迭代次数
        dataset_name = "indian"  # 数据集名称
        pass
    if FLAG == 2:
        data_mat = sio.loadmat('./Datasets/PaviaU/PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('./Datasets/PaviaU/PaviaU_gt.mat')
        gt = gt_mat['paviaU_gt']

        # 参数预设
        train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
        val_ratio = 0.01 # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 9  # 样本类别数
        learning_rate = 6e-4 # 学习率
        max_epoch = 1000  # 迭代次数
        weight_decay = 2e-5
        split_height = 2
        split_width = 2
        EDGE=5
        dataset_name = "paviaU_SSCDenseNet"  # 数据集名称
        pass
     ###########
    train_samples_per_class=curr_train_ratio#当定义为每类样本个数时,则该参数更改为训练样本数
    val_samples=class_count


    train_ratio=curr_train_ratio

    cmap = cm.get_cmap('jet', class_count + 1)
    plt.set_cmap(cmap)
    m, n, d = data.shape  # 高光谱数据的三个维度
    n_bands=d

    data = np.reshape(data, [m * n, d])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [m, n, d])
    #打印每类样本个数
    # gt_reshape=np.reshape(gt, [-1])
    # for i in range(class_count):
    #     idx = np.where(gt_reshape == i + 1)[-1]
    #     samplesCount = len(idx)
    #     print(samplesCount)

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
                rand_idx = random.sample(rand_list, np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
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
                if real_train_samples_per_class>samplesCount:
                    #real_train_samples_per_class=samplesCount
                    real_train_samples_per_class=int(train_samples_per_class/2)
                    # val_samples_per_class=0
                rand_idx = random.sample(rand_list,
                                         real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
                rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                train_rand_idx.append(rand_real_idx_per_class_train)
                # if val_samples_per_class>0:
                #     rand_real_idx_per_class_val = idx[rand_idx[-val_samples_per_class:]]
                #     val_rand_idx.append(rand_real_idx_per_class_val)
            train_rand_idx = np.array(train_rand_idx)
            val_rand_idx = np.array(val_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)


            train_data_index = set(train_data_index)
            # val_data_index = set(val_data_index)
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

        # 将数据扩展一维，以满足网络输入需求
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


    model = CCCon2B.SSCDNonLModel(class_count, n_bands, 200) # Criss Cross Model CCNet 2B parallel

    print(model)
    if use_cuda: torch.backends.cudnn.benchmark = True
    if use_cuda: model.cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-5)
    print('lr: ',learning_rate, '  weight_dacay: ', weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-4, nesterov=True)

    epoch=1000
    best_acc = -1

    for eep in range(epoch):
        for batch_idx, (inputs, labels) in enumerate(trainloader):#batch_idx是enumerate（）函数自带的索引，从0开始
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
            optimizer.zero_grad()
            #output, _, _ = model(inputs)
            output= model(inputs)
            #loss = F.nll_loss(output, labels.long())
            #print(output.shape,  labels.shape)
            loss=criterion(output, labels.long())
            optimizer.zero_grad()   # 所有参数的梯度清零
            loss.backward()         #即反向传播求梯度
            optimizer.step()        #调用optimizer进行梯度下降更新参数
            #print(loss.data)
            #aa, countt=accuracy(output.data, labels.data, class_count+1)
            #print(eep, aa, countt, aa/countt)
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
            #Testoutput, attention_1, attention_2 = model(Testinputs)
            OutputWhole = PatchStack(Output, m, n, patch_height, patch_width, split_height, split_width, EDGE, class_count+1)
            #Testoutput = model(Testinputs)
            #Taa, Tcountt=accuracy(Testoutput.data, Testtargets.data, class_count+1)
            AC, OA, AA, rightNum, testNum= ClassificationAccuracy(OutputWhole, Test_Label, class_count+1)
            kappa = Kappa(OutputWhole, Test_Label, class_count+1)
            print("eep", eep, " test", rightNum, testNum, "OA", OA, "AA", AA, "kappa", kappa)
            print(OA, AA, kappa, AC)
        if eep==400:
            OA=np.round(OA*100, decimals=2)
            OutputWhole = PatchStack(Output, m, n, patch_height, patch_width, split_height, split_width, EDGE, class_count+1)
            Draw_Classification_Map(OutputWhole, 'ResultsImage/' + dataset_name + '_NL_' + str(train_ratio) + '_' + str(OA))


        if loss.data<=0.00005:
            break


    model.train()
    model.eval()









import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import MSSGU
from utils import Draw_Classification_Map,get_Samples_GT,GT_To_One_Hot
from SegmentMap import SegmentMap
import h5py
from scipy import misc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu" )

# FLAG =1, indian
# FLAG =2, paviaU
# FLAG =3, salinas
# samples_type = ['ratio', 'same_num'][0]

# for (FLAG, curr_train_ratio) in [(1,5),(1,10),(1,15),(1,20),(1,25),
# (2,5),(2,10),(2,15),(2,20),(2,25),
# (3,5),(3,10),(3,15),(3,20),(3,25)]:#(2,0.01),(3,0.01) (1,0.1),(2,0.01),(3,0.01)
# # for (FLAG, curr_train_ratio) in [(1,25),(3,25)]:

# for (FLAG, curr_train_ratio,Scale) in [(1,0.1,100),(1,0.1,200),(1,0.1,300),(1,0.1,400),(1,0.1,500),
# (2,0.01,100),(2,0.01,200),(2,0.01,300),(2,0.01,400),(2,0.01,500),
# (3,0.01,100),(3,0.01,200),(3,0.01,300),(3,0.01,400),(3,0.01,500)]:
# for (FLAG, curr_train_ratio,Scale) in [(3,5,100),(3,25,100)]:
for Neighbors in [0]: #0, 5,10,15,20
    # for (FLAG, curr_train_ratio,Unet_Depth) in [ (1,0.05,1),(1,0.05,2),(1,0.05,3),(1,0.05,4),
    #                                              (2,0.005,1),(2,0.005,2),(2,0.005,3),(2,0.005,4),
    #                                              (3,0.005,1),(3,0.005,2),(3,0.005,3),(3,0.005,4)]:
    # for (FLAG, curr_train_ratio,Unet_Depth) in [ (1,5,4),(1,10,4),(1,15,4),(1,20,4),(1,25,4),
    #                                              (2,5,4),(2,10,4),(2,15,4),(2,20,4),(2,25,4),
    #                                              (3,5,4),(3,10,4),(3,15,4),(3,20,4),(3,25,4)]:
    for (FLAG, curr_train_ratio,Unet_Depth) in [(1,0.05,4)]: #,(2,0.005,4),(3,0.005,4)
    # for (FLAG, curr_train_ratio, Unet_Depth) in [(3,0.005,4)]: #visualize(1, 0.05, 4),(2,0.005,4),
        # NOTE THAT the Unet_Depth here denotes the depth of graphs only, i.e., the real network depth is equal to (Unet_Depth+1)
        torch.cuda.empty_cache()
        OA_ALL = [];AA_ALL = [];KPP_ALL = [];AVG_ALL = [];Train_Time_ALL=[];Test_Time_ALL=[]
        samples_type = 'ratio' if curr_train_ratio < 1 else 'same_num'
    
        # Seed_List=[0,1,2,3,4,5,6,7,8,9]
        # Seed_List=[0,1,2,3,4]
        Seed_List = [0,]

        if FLAG == 1:
            data_mat = sio.loadmat('..\\HyperImage_data\\indian\\Indian_pines_corrected.mat')
            data = data_mat['indian_pines_corrected']
            gt_mat = sio.loadmat('..\\HyperImage_data\\indian\\Indian_pines_gt.mat')
            gt = gt_mat['indian_pines_gt']
        
            # 参数预设
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 16  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 600  # 迭代次数
            dataset_name = "indian_"  # 数据集名称
            split_height = 1
            split_width = 1
            pass
        if FLAG == 2:
            data_mat = sio.loadmat('..\\HyperImage_data\\paviaU\\PaviaU.mat')
            data = data_mat['paviaU']
            gt_mat = sio.loadmat('..\\HyperImage_data\\paviaU\\Pavia_University_gt.mat')
            gt = gt_mat['pavia_university_gt']
        
            # 参数预设
            val_ratio = 0.005  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 9  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 600  # 迭代次数
            dataset_name = "paviaU_"  # 数据集名称
            split_height = 1
            split_width = 1
            pass
        if FLAG == 3:
            data_mat = sio.loadmat('..\\HyperImage_data\\Salinas\\Salinas_corrected.mat')
            data = data_mat['salinas_corrected']
            gt_mat = sio.loadmat('..\\HyperImage_data\\Salinas\\Salinas_gt.mat')
            gt = gt_mat['salinas_gt']
        
            # 参数预设
            val_ratio = 0.005  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 16  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 600  # 迭代次数
            dataset_name = "salinas_"  # 数据集名称
            split_height = 1
            split_width = 1
            pass
        if FLAG == 4:
            data_mat = sio.loadmat('..\\HyperImage_data\\KSC\\KSC.mat')
            data = data_mat['KSC']
            gt_mat = sio.loadmat('..\\HyperImage_data\\KSC\\KSC_gt.mat')
            gt = gt_mat['KSC_gt']
        
            # 参数预设
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 13  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 600  # 迭代次数
            dataset_name = "KSC_"  # 数据集名称
            split_height = 1
            split_width = 1
            pass
        if FLAG == 5:
            data_mat = sio.loadmat('..\\HyperImage_data\\Houston2013\\Houston.mat')
            data = data_mat['Houston']
            gt_mat = sio.loadmat('..\\HyperImage_data\\Houston2013\\Houston_GT_15.mat')
            gt = gt_mat['Houston_GT']
        
            # 参数预设
            # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            # class_count = 15  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 600  # 迭代次数
            dataset_name = "Houston_"  # 数据集名称
            pass
        if FLAG == 6:
            data_mat = sio.loadmat('..\\HyperImage_data\\HyRANK\\Loukia.mat')
            data = data_mat['Loukia']
            gt_mat = sio.loadmat('..\\HyperImage_data\\HyRANK\\Loukia_GT.mat')
            gt = gt_mat['Loukia_GT']
        
            # 参数预设
            # train_ratio = 0.01  # 训练集比例。注意，训练集为按照‘每类’随机选取
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 14  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 600  # 迭代次数
            dataset_name = "Loukia_"  # 数据集名称
            split_height = 1
            split_width = 1
            pass
        if FLAG == 7:
            data_mat = sio.loadmat('..\\HyperImage_data\\Botswana\\Botswana.mat')
            data = data_mat['Botswana']
            gt_mat = sio.loadmat('..\\HyperImage_data\\Botswana\\Botswana_gt.mat')
            gt = gt_mat['Botswana_gt']
        
            # 参数预设
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 14  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 600  # 迭代次数
            dataset_name = "Botswana_"  # 数据集名称
            split_height = 1
            split_width = 2
            pass
        if FLAG == 8:
            # data_mat = sio.loadmat('..\\HyperImage_data\\Houston2018\\HoustonU.mat')
            data_mat = h5py.File('..\\HyperImage_data\\Houston2018\\HoustonU.mat')
            data = data_mat['houstonU']
            data = np.transpose(data, [2, 1, 0])
            # gt_mat = sio.loadmat('..\\HyperImage_data\\Houston2018\\HoustonU_gt.mat')
            gt_mat = h5py.File('..\\HyperImage_data\\Houston2018\\HoustonU_gt.mat')
            gt = gt_mat['houstonU_gt']
            gt = np.transpose(gt, [1, 0])
            # 参数预设
            val_ratio = 0.005  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 20  # 样本类别数
            learning_rate = 1e-4  # 学习率
            max_epoch = 600  # 迭代次数
            dataset_name = "HoustonU_"  # 数据集名称
            pass
        if FLAG == 9:
            # data_mat = sio.loadmat('..\\HyperImage_data\\Houston2018\\HoustonU.mat')
            data_mat = h5py.File('..\\HyperImage_data\\Houston2018\\HoustonU.mat')
            data = data_mat['houstonU']
            data = np.transpose(data, [2, 1, 0])
            # gt_mat = sio.loadmat('..\\HyperImage_data\\Houston2018\\HoustonU_gt.mat')
            gt_mat = h5py.File('..\\HyperImage_data\\Houston2018\\HoustonU_gt.mat')
            gt = gt_mat['houstonU_gt']
            gt = np.transpose(gt, [1, 0])

            # 参数预设
            val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = len(set(np.reshape(gt,[-1])))-1  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 600  # 迭代次数
            dataset_name = "HoustonU_"  # 数据集名称
            pass
        if FLAG == 10:
            data_mat = sio.loadmat('..\\HyperImage_data\\xuzhou\\xuzhou.mat')
            data = data_mat['xuzhou']
            gt_mat = sio.loadmat('..\\HyperImage_data\\xuzhou\\xuzhou_gt.mat')
            gt = gt_mat['xuzhou_gt']
            # 参数预设
            val_ratio = 0.001  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 9  # 样本类别数
            learning_rate = 5e-4  # 学习率
            max_epoch = 600  # 迭代次数
            dataset_name = "xuzhou_"  # 数据集名称
            pass
        ###########
        if samples_type == 'same_num': val_ratio = 10# 当定义为每类样本个数时,则该参数更改为训练样本数
        
        train_ratio = curr_train_ratio
        cmap = cm.get_cmap('jet', class_count + 1)
        plt.set_cmap(cmap)
        m, n, d = data.shape  # 高光谱数据的三个维度
    
        # 数据standardization标准化,即提前全局BN
        orig_data=data
        height, width, bands = data.shape  # 原始高光谱数据的三个维度
        data = np.reshape(data, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        data = np.reshape(data, [height, width, bands])
        
        
        # 打印每类样本个数
        gt_reshape=np.reshape(gt, [-1])
        samplesCount_list = []
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            samplesCount_list.append(samplesCount)
        print(samplesCount_list)
        
        for curr_seed in Seed_List:
            train_samples_gt, test_samples_gt, val_samples_gt= get_Samples_GT(curr_seed, gt, class_count, curr_train_ratio,val_ratio, samples_type )
            train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
            test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
            val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)
            train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
            test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
            val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)
            Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图
            
            
            # 打印训练 验证 测试样本数量
            train_val_test_gt=[train_samples_gt,val_samples_gt,test_samples_gt]
            for type in range(3):
                gt_reshape = np.reshape(train_val_test_gt[type], [-1])
                print("===============================")
                samplesCount_list=[]
                for i in range(class_count):
                    idx = np.where(gt_reshape == i + 1)[-1]
                    samplesCount = len(idx)
                    samplesCount_list.append(samplesCount)
                print(samplesCount_list)
            
    
            ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
            # 训练集
            train_label_mask = np.zeros([m * n, class_count])
            temp_ones = np.ones([class_count])
            train_samples_gt = np.reshape(train_samples_gt, [m * n])
            for i in range(m * n):
                if train_samples_gt[i] != 0:
                    train_label_mask[i] = temp_ones
            train_label_mask = np.reshape(train_label_mask, [m* n, class_count])
    
            # 测试集
            test_label_mask = np.zeros([m * n, class_count])
            temp_ones = np.ones([class_count])
            test_samples_gt = np.reshape(test_samples_gt, [m * n])
            for i in range(m * n):
                if test_samples_gt[i] != 0:
                    test_label_mask[i] = temp_ones
            test_label_mask = np.reshape(test_label_mask, [m* n, class_count])
    
            # 验证集
            val_label_mask = np.zeros([m * n, class_count])
            temp_ones = np.ones([class_count])
            val_samples_gt = np.reshape(val_samples_gt, [m * n])
            for i in range(m * n):
                if val_samples_gt[i] != 0:
                    val_label_mask[i] = temp_ones
            val_label_mask = np.reshape(val_label_mask, [m* n, class_count])

            ##################### 产生分层分配矩阵S和邻接矩阵A
            data_for_segment = np.reshape(data, [height, width, -1])
            tic = time.clock()
            # ST = SegmentTree(data_for_segment,np.reshape(train_samples_gt,[height,width]))
            # ST = SegmentTree(data_for_segment,np.zeros_like(gt))
            # segmentTree = ST.createSegmentTree()
            # S_list, A_list = ST.getHierarchy(hierarchy)
            toc = time.clock()
            HierarchicalSegmentation_Time = toc - tic
            print('segment tree -- cost time:', toc - tic)
    
            tic = time.clock()
            
            SM = SegmentMap(dataset_name)
            S_list, A_list = SM.getHierarchy()
            S_list=S_list[0:int(Unet_Depth)]
            A_list=A_list[0:int(Unet_Depth)]
            
            # 输出每层的最小-最大邻居数量
            for i in range(Unet_Depth):
                a=A_list[i].sum(-1)
                print(a.min(),a.max(),a.mean())
            
            toc = time.clock()
            HierarchicalSegmentation_Time=HierarchicalSegmentation_Time+(toc - tic)
            print('getHierarchy -- cost time:', toc - tic)
            
            # GPU
            S_list_gpu=[]
            A_list_gpu=[]
            for i in range(len(S_list)):
                S_list_gpu.append(torch.from_numpy(np.array( S_list[i],dtype=np.float32 )).to(device))
                A_list_gpu.append(torch.from_numpy(np.array( A_list[i],dtype=np.float32 )).to(device))
                
            #转到GPU
            train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
            test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
            val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
            #转到GPU
            train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
            test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
            val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
            #转到GPU
            train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
            test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
            val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)
            
            
            # net_input=np.transpose(data,[2,0,1])
            net_input=np.array( data,np.float32)
            net_input=torch.from_numpy(net_input.astype(np.float32)).to(device)
            net = MSSGU.HiGCN(height, width, bands, class_count, S_list_gpu, A_list_gpu, Neighbors).to(device)
            
            total_param = sum([param.nelement() for param in net.parameters()])
            print("Number of parameter: %.2fK" % (total_param / 1e3))
            # print("parameters", net.parameters(), len(list(net.parameters())))
            # flops, params = profile(net, inputs=(net_input,))
            # print(flops)
            # print(params)
            
            def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
                real_labels = reallabel_onehot
                # print("real_labels", torch.sum(real_labels))
        
                ##  加权交叉熵损失函数
                we = -torch.mul(real_labels,torch.log(predict+1e-15))
                we = torch.mul(we, reallabel_mask)
    
                # # #加权##################
                # we2 = torch.sum(real_labels, 0)   # 每类训练样本个数 加1是为了防止除0
                # we2 = 1. / (we2+ 1)  # 每类样本的权重
                # we2 = torch.unsqueeze(we2, 0)
                # we2 = we2.repeat([m * n, 1])
                # we = torch.mul(we, we2)
                pool_cross_entropy = torch.sum(we)
                return pool_cross_entropy #+ 0.01*reconstruction_loss
            
            # output= net(net_input)
            zeros = torch.zeros([m * n]).to(device).float()
            def evaluate_performance(network_output,train_samples_gt,train_samples_gt_onehot, require_AA_KPP=False,printFlag=True):
                if False==require_AA_KPP:
                    with torch.no_grad():
                        available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
                        available_label_count=available_label_idx.sum()#有效标签的个数
                        correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
                        OA= correct_prediction.cpu()/available_label_count
                        return OA
                else:
                    with torch.no_grad():
                        #计算OA
                        available_label_idx=(train_samples_gt!=0).float()#有效标签的坐标,用于排除背景
                        available_label_count=available_label_idx.sum()#有效标签的个数
                        correct_prediction =torch.where(torch.argmax(network_output, 1) ==torch.argmax(train_samples_gt_onehot, 1),available_label_idx,zeros).sum()
                        OA= correct_prediction.cpu()/available_label_count
                        OA=OA.cpu().numpy()
                        
                        # 计算AA
                        zero_vector = np.zeros([class_count])
                        output_data=network_output.cpu().numpy()
                        train_samples_gt=train_samples_gt.cpu().numpy()
                        train_samples_gt_onehot=train_samples_gt_onehot.cpu().numpy()
                        
                        output_data = np.reshape(output_data, [m * n, class_count])
                        idx = np.argmax(output_data, axis=-1)
                        for z in range(output_data.shape[0]):
                            if ~(zero_vector == output_data[z]).all():
                                idx[z] += 1
                        # idx = idx + train_samples_gt
                        count_perclass = np.zeros([class_count])
                        correct_perclass = np.zeros([class_count])
                        for x in range(len(train_samples_gt)):
                            if train_samples_gt[x] != 0:
                                count_perclass[int(train_samples_gt[x] - 1)] += 1
                                if train_samples_gt[x] == idx[x]:
                                    correct_perclass[int(train_samples_gt[x] - 1)] += 1
                        test_AC_list = correct_perclass / count_perclass
                        test_AA = np.average(test_AC_list)
    
                        # 计算KPP
                        test_pre_label_list = []
                        test_real_label_list = []
                        output_data = np.reshape(output_data, [m * n, class_count])
                        idx = np.argmax(output_data, axis=-1)
                        idx = np.reshape(idx, [m, n])
                        for ii in range(m):
                            for jj in range(n):
                                if Test_GT[ii][jj] != 0:
                                    test_pre_label_list.append(idx[ii][jj] + 1)
                                    test_real_label_list.append(Test_GT[ii][jj])
                        test_pre_label_list = np.array(test_pre_label_list)
                        test_real_label_list = np.array(test_real_label_list)
                        kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                          test_real_label_list.astype(np.int16))
                        test_kpp = kappa
    
                        # 输出
                        if printFlag:
                            print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                            print('acc per class:')
                            print(test_AC_list)
    
                        OA_ALL.append(OA)
                        AA_ALL.append(test_AA)
                        KPP_ALL.append(test_kpp)
                        AVG_ALL.append(test_AC_list)
                        
                        
                        
                        # 保存数据信息
                        f = open('results\\' + dataset_name + '_results.txt', 'a+')
                        str_results = '\n======================' \
                                      + " learning rate=" + str(learning_rate) \
                                      + " epochs=" + str(max_epoch) \
                                      + " train ratio=" + str(train_ratio) \
                                      + " val ratio=" + str(val_ratio) \
                                      + " ======================" \
                                      + "\nOA=" + str(OA) \
                                      + "\nAA=" + str(test_AA) \
                                      + '\nkpp=' + str(test_kpp) \
                                      + '\nacc per class:' + str(test_AC_list) + "\n"
                                      # + '\ntrain time:' + str(time_train_end - time_train_start) \
                                      # + '\ntest time:' + str(time_test_end - time_test_start) \
                        f.write(str_results)
                        f.close()
                        return OA
                    
            # evaluate_performance(output,train_samples_gt,train_samples_gt_onehot)
            optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=0.0001)#,weight_decay=0.0001
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.8, patience=2,  verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    
            # 训练
            best_loss=99999
            best_OA=0
            Stop_Flag=0
            net.train()
            tic1 = time.clock()
            for i in range(max_epoch+1):
                optimizer.zero_grad()  # zero the gradient buffers
                output,_,_ = net(net_input)
                loss = compute_loss(output,train_samples_gt_onehot,train_label_mask)
                loss.backward(retain_graph=False)
                optimizer.step()  # Does the update
                if i%10==0:
                    with torch.no_grad():
                        net.eval()
                        output,_,_ = net(net_input,showFlag=True)
                        trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                        trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                        valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                        valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                        testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot)
                        print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}\t test OA={}".format(str(i + 1), trainloss, trainOA, valloss, valOA,testOA))
    
                        if valloss < best_loss or valOA>best_OA:
                            best_loss = valloss
                            best_OA=valOA
                            Stop_Flag=0
                            torch.save(net.state_dict(),"model\\best_model.pt")
                            print('save model...')

                    torch.cuda.empty_cache()
                    net.train()
            toc1 = time.clock()
            print("\n\n====================training done. starting evaluation...========================\n")
            training_time=toc1 - tic1 + HierarchicalSegmentation_Time #分割耗时需要算进去
            Train_Time_ALL.append(training_time)
            
            torch.cuda.empty_cache()
            with torch.no_grad():
                net.load_state_dict(torch.load("model\\best_model.pt"))
                net.eval()
                tic2 = time.clock()
                # output,A_encoders, A_decoders = net(net_input)
                output,encoder_features, decoder_features = net(net_input) ###############
                toc2 = time.clock()
                testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
                testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot,require_AA_KPP=True,printFlag=False)
                print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
                #计算
                classification_map=torch.argmax(output, 1).reshape([height,width]).cpu()+1
                Draw_Classification_Map(classification_map,"results\\"+dataset_name+str(testOA))
                testing_time=toc2 - tic2 + HierarchicalSegmentation_Time #分割耗时需要算进去
                Test_Time_ALL.append(testing_time)
                # sio.savemat(dataset_name+"softmax",{'softmax':output.reshape([height,width,-1]).cpu().numpy()})
                
                # 用户图边可视化
                # np.savez(dataset_name+"A_encoders",
                #          a1=A_encoders[0].cpu().numpy(),
                #          a2=A_encoders[1].cpu().numpy(),
                #          a3=A_encoders[2].cpu().numpy(),
                #          a4=A_encoders[3].cpu().numpy())
                # np.savez(dataset_name+"A_decoders",
                #          a1=A_decoders[0].cpu().numpy(),
                #          a2=A_decoders[1].cpu().numpy(),
                #          a3=A_decoders[2].cpu().numpy(),
                #          a4=A_decoders[3].cpu().numpy())
                
                # # 计算特征相关性
                # test_gt_reshape=train_samples_gt.detach().cpu().numpy()
                # for i in range(4):
                #     df = decoder_features[-1 - i].detach().cpu().numpy()
                #     df_last = decoder_features[-2 - i]
                #     df_last=torch.mm(S_list_gpu[i], df_last).detach().cpu().numpy()
                #     # df_last=torch.cat([df_last,df_last],dim=-1)
                #     ed = encoder_features[i].detach().cpu().numpy()
                #     if i==0: idx = np.where(test_gt_reshape>0)[0]
                #     else:
                #         test_gt_reshape = np.matmul(S_list_gpu[i - 1].detach().cpu().numpy().transpose(), test_gt_reshape)
                #         idx = np.where(test_gt_reshape > 0)[0]
                #     df = df[idx, :]
                #     df_last = df_last[idx, :]
                #     ed = ed[idx, :]
                #     co1 = dcor.distance_correlation(df, ed)
                #     co2 = dcor.distance_correlation(df, df_last)
                #     print(co1, co2)

                # # 只计算与Final Feature Map的相关性
                # test_gt_reshape = train_samples_gt.detach().cpu().numpy()
                # # test_gt_reshape = np.zeros_like(test_gt_reshape)
                # # for i in range(test_gt_reshape.shape[0]):
                # #     if random.random() < 0.10: test_gt_reshape[i] = 1
                #
                # final_feature_map = decoder_features[-1].detach().cpu().numpy()
                # for i in range(4):
                #     df_last = decoder_features[-2 - i]
                #     df_last = torch.mm(S_list_gpu[i], df_last).detach().cpu().numpy()
                #     ed = encoder_features[i].detach().cpu().numpy()
                #     if i == 0:
                #         idx = np.where(test_gt_reshape > 0)[0]
                #     else:
                #         test_gt_reshape = np.matmul(S_list_gpu[i - 1].detach().cpu().numpy().transpose(),
                #                                     test_gt_reshape)
                #         idx = np.where(test_gt_reshape > 0)[0]
                #         final_feature_map=np.matmul(S_list_gpu[i - 1].detach().cpu().numpy().transpose(),
                #                                     final_feature_map)
                #
                #     df = final_feature_map[idx, :]
                #     df_last = df_last[idx, :]
                #     ed = ed[idx, :]
                #     co1 = dcor.distance_correlation(df, ed)
                #     co2 = dcor.distance_correlation(df, df_last)
                #     print(co1, co2)

                # #############################################################
                # # 将特征映射到每一类上,可视化
                # def restoreOutputShape(featureMap,hierarchyIdx):
                #     x=featureMap
                #     for i in range(hierarchyIdx): x = torch.matmul(S_list_gpu[hierarchyIdx - i-1], x)
                #     return x
                # curr_gt = train_samples_gt.detach().cpu().numpy()
                # from sklearn.linear_model import LogisticRegression
                # from scipy.special import softmax
                #
                # for i in range(len(encoder_features)):
                #     F=encoder_features[i]
                #     F=restoreOutputShape(F,i)
                #     F=F.detach().cpu().numpy()
                #     train_s=F[np.where(curr_gt>0)[0]]
                #     train_l=curr_gt[np.where(curr_gt>0)[0]]
                #     lr_model = LogisticRegression()  # 调用模型，但是并未经过任何调参操作，使用默认值
                #     lr_model.fit(train_s, train_l)  # 训练模型
                #     Y=lr_model.decision_function(F)
                #     Y=softmax(Y,axis=-1)
                #     Y=np.reshape(Y,[height,width,-1])
                #     for j in range(class_count):
                #         plt.figure()
                #         plt.imshow(Y[:,:,j],cmap = 'gray')
                #         plt.axis('off')
                #         plt.savefig("featuremap_of_coder\\" + dataset_name+"_class_"+str(j+1)+"_encoder_"+str(i+1)+".jpeg", dpi=300,bbox_inches = 'tight',pad_inches = 0)
                # for i in range(len(decoder_features)):
                #     F=decoder_features[4-i]
                #     F=restoreOutputShape(F,i)
                #     F=F.detach().cpu().numpy()
                #     train_s=F[np.where(curr_gt>0)[0]]
                #     train_l=curr_gt[np.where(curr_gt>0)[0]]
                #     lr_model = LogisticRegression()  # 调用模型，但是并未经过任何调参操作，使用默认值
                #     lr_model.fit(train_s, train_l)  # 训练模型
                #     Y=lr_model.decision_function(F)
                #     Y=softmax(Y,axis=-1)
                #     Y=np.reshape(Y,[height,width,-1])
                #     for j in range(class_count):
                #         plt.figure()
                #         plt.imshow(Y[:,:,j],cmap = 'gray')
                #         plt.axis('off')
                #         plt.savefig("featuremap_of_coder\\" + dataset_name+"_class_"+str(j+1)+"_decoder_"+str(i+1)+".jpeg", dpi=300,bbox_inches = 'tight',pad_inches = 0)

                

            torch.cuda.empty_cache()
            del net
            
        OA_ALL = np.array(OA_ALL)
        AA_ALL = np.array(AA_ALL)
        KPP_ALL = np.array(KPP_ALL)
        AVG_ALL = np.array(AVG_ALL)
        Train_Time_ALL=np.array(Train_Time_ALL)
        Test_Time_ALL=np.array(Test_Time_ALL)
    
        print("\nTrain ratio={}, Neighbors={}, Depth={}".format(curr_train_ratio,Neighbors,Unet_Depth),
              "\n==============================================================================")
        print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
        print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
        print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
        print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
        print("Average training time:{}".format(np.mean(Train_Time_ALL)))
        print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
        
        # 保存数据信息
        f = open('results\\' + dataset_name + '_results.txt', 'a+')
        str_results = '\n\n************************************************' \
        +"\nTrain ratio={}, Neighbors={}, Depth={}".format(curr_train_ratio,Neighbors,Unet_Depth) \
        +'\nOA='+ str(np.mean(OA_ALL))+ '+-'+ str(np.std(OA_ALL)) \
        +'\nAA='+ str(np.mean(AA_ALL))+ '+-'+ str(np.std(AA_ALL)) \
        +'\nKpp='+ str(np.mean(KPP_ALL))+ '+-'+ str(np.std(KPP_ALL)) \
        +'\nAVG='+ str(np.mean(AVG_ALL,0))+ '+-'+ str(np.std(AVG_ALL,0)) \
        +"\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
        +"\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
        f.write(str_results)
        f.close()
        

    
    
    
    
    
    
    

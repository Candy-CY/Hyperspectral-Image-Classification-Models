from HSIC_FM import FullModel
from func import *
import os
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from network import *
import sys
import argparse
from thop import profile


## 指定显卡
USE_GPU=True
if USE_GPU:
    GPU_ID = "2"  # according to GPU_ID
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
else:
    device=torch.device('cpu')

print('GPU device:', GPU_ID)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")

    argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
    argparser.add_argument("--gru", action='store_true', help="whether to use gru")
    argparser.add_argument("--vanillarnn", action='store_true', help="whether to use vanilla rnn")

    argparser.add_argument("--cudnn", action='store_true', help="whether to use cuda")
    argparser.add_argument("--use_cuda", action='store_false', help="cuda or not")

    argparser.add_argument("--dataset", type=str, default="indian", help="which dataset")
    # argparser.add_argument("--numseq", type=int, default=-1)  # if numseq -1 numseq = numbands
    argparser.add_argument("--numseq", type=int, default=1)
    argparser.add_argument("--pca", type=int, default=-1)

    argparser.add_argument("--use_val", default=False, action='store_true', help="validation or not")
    argparser.add_argument("--valpercent", type=float, default=0.1)

    argparser.add_argument("--random_state", type=int, default=69)
    argparser.add_argument("--batch_size", type=int, default=100)
    argparser.add_argument("--epochs", type=int, default=200)
    argparser.add_argument("--idtest", type=int, default=0)
    argparser.add_argument("--d", type=int, default=64)
    # argparser.add_argument("--dropout", type=float, default=0.10)
    argparser.add_argument("--tpercent", type=float, default=-1)  # if -1, 15 indian 10 the others
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.001)

    args = argparser.parse_args()

    torch.backends.cudnn.enabled = True if args.cudnn else False

#################################### load dataset #######################################
a=load()

dataID = 6
NUM_CLASSES = 13
StorageLocation = './Result/'


if not os.path.isdir(StorageLocation):
    os.makedirs(StorageLocation)

#实验次数
Experiment_num = 2
if dataID == 1:
    All_data, labeled_data, rows_num, categories, r, c, FLAG = a.load_data(flag='pavia')
    epoch = 140
    batchsize = 128
elif dataID == 2:
    All_data, labeled_data, rows_num, categories, r, c, FLAG = a.load_data(flag='indian')
    epoch = 135
    batchsize = 128
elif dataID == 6:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='ksc')
    epoch = 200
    batchsize = 512
elif dataID == 7:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='houston')
    epoch = 165
    batchsize = 128
elif dataID == 4:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='sali')
    epoch = 200
elif dataID == 8:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='hanchuan')
    epoch = 400
    batchsize = 128
elif dataID == 9:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='honghu')
    epoch = 85
    batchsize = 128
elif dataID == 10:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='longkou')
    epoch = 85
    batchsize = 128

print('Data has been loaded successfully!')


# 设置归一化范围
mi = -0.5
ma = 0.5
a=preprocess('pca')
All_data_Spe = All_data
Alldata_DR=a.Dim_reduction(All_data)  # pc: 1
a = product(c, FLAG)
# All_data_norm=a.normlization(Alldata_DR[:,1:-1],mi,ma)
All_data_norm=a.normlization(Alldata_DR,mi,ma)
All_data_Spe=a.normlization(All_data_Spe[:,1:-1],mi,ma)
image_data3D=All_data_norm.reshape(r,c,-1)
All_data_Spe=All_data_Spe.reshape(r,c,-1)

##################### 数据准备 ####################
half_s = 13

image_3d_lr=np.fliplr(image_data3D)
image_3d_ud=np.flipud(image_data3D)
image_3d_corner=np.fliplr(np.flipud(image_data3D))

image_3d_temp1=np.hstack((image_3d_corner,image_3d_ud,image_3d_corner))
image_3d_temp2=np.hstack((image_3d_lr,image_data3D,image_3d_lr))
image_3d_merge=np.vstack((image_3d_temp1,image_3d_temp2,image_3d_temp1))

image_3d_mat_origin=image_3d_merge[(r-half_s):2*r+half_s,(c-half_s):2*c+half_s,:]

All_data_Spe_lr=np.fliplr(All_data_Spe)
All_data_Spe_ud=np.flipud(All_data_Spe)
All_data_Spe_corner=np.fliplr(np.flipud(All_data_Spe))

All_data_Spe_temp1=np.hstack((All_data_Spe_corner,All_data_Spe_ud,All_data_Spe_corner))
All_data_Spe_temp2=np.hstack((All_data_Spe_lr,All_data_Spe,All_data_Spe_lr))
All_data_Spe_merge=np.vstack((All_data_Spe_temp1,All_data_Spe_temp2,All_data_Spe_temp1))

All_data_Spe_mat_origin=All_data_Spe_merge[(r-half_s):2*r+half_s,(c-half_s):2*c+half_s,:]


del image_3d_lr,image_3d_ud,image_3d_corner,image_3d_temp1,image_3d_temp2,image_3d_merge
del All_data_Spe_lr,All_data_Spe_ud,All_data_Spe_corner,All_data_Spe_temp1,All_data_Spe_temp2,All_data_Spe_merge

#################################### 空间数据，训练、检验、测试、预测 ###########################
#生成训练样本
Experiment_result=np.zeros([categories+5, Experiment_num + 2])

for count in range(0, Experiment_num):
    a = product(c, FLAG)
    rows_num,trn_num,tes_num,pre_num=a.generation_num(labeled_data,rows_num,All_data)

    #################################### Training #####################################
    ############ label #############
    y_trn = All_data[trn_num, -1]
    trn_YY = torch.from_numpy(y_trn - 1)

    ############ data  #############
    trn_spe, trn_spat,trn_num,_=a.production_data_trn_SpeAll(rows_num,trn_num,half_s,image_3d_mat_origin, All_data_Spe_mat_origin)

    trn_spat = trn_spat[:,:,:,:]
    trn_spe = trn_spe[:,:,np.newaxis]

    np.save(StorageLocation + repr(dataID) + '_trn_num' + '.npy', trn_num)
    np.save(StorageLocation + repr(dataID) + '_pre_num' + '.npy', pre_num)
    np.save(StorageLocation + repr(dataID) + '_y_trn' + '.npy', y_trn)
    np.save(StorageLocation + repr(dataID) + '_image_3d_mat_origin' + '.npy', image_3d_mat_origin)


    trn_XX_spat = torch.from_numpy(trn_spat.transpose(0, 3, 1, 2))
    trn_XX_spe = torch.from_numpy(trn_spe.transpose(0, 2, 1))


    del trn_spat, trn_spe

    torch.cuda.empty_cache()

    trn_dataset=TensorDataset(trn_XX_spe, trn_XX_spat, trn_YY)
    trn_loader=DataLoader(trn_dataset,batch_size=batchsize
                          ,sampler=SubsetRandomSampler(range(trn_XX_spat.shape[0])))

    net = FullModel(
        num_classes = NUM_CLASSES,
        dims = (64, 128, 256, 512),
        depths = (3, 4, 1, 1),
        mhsa_types = ('l', 'l', 'l', 'g'),
        args = args,
        input_size = 1,
        bands = trn_XX_spe.shape[2]
    )

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.cuda()
    torch.set_num_threads(1)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3/10, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)


    ##### FLOPS, PARAMS #####
    flops, params = profile(net, (trn_XX_spe.float().cuda(), trn_XX_spat.float().cuda(),))
    print('FLOPs: {} G, PARAMS: {} M'.format(flops/1e9, params/1e6))
    #########################


    loss_trn = []
    acc_trn = []
    tsne_trn = []
    tsne_trn_y = []

    runtime1 = time.time()
    trn_time1 = time.time()
    for i in range(1, epoch+1):
        b=operate()
        loss_trn, acc_trn, tsne_trn, tsne_trn_y = b.train(i, loss_trn, acc_trn, tsne_trn, tsne_trn_y, net, optimizer, scheduler, trn_loader, criterion)
    trn_time2 = time.time()

    ######## output loss, acc ########
    plt.figure(1)
    plt.plot(np.array(loss_trn), label='Training')
    plt.legend()
    plt.savefig(StorageLocation + str(FLAG) + '_loss_trn' + '.png')
    # plt.show()

    plt.figure(2)
    plt.plot(np.array(acc_trn), label='Training')
    plt.legend()
    plt.savefig(StorageLocation + str(FLAG) + '_acc_trn' + '.png')
    # plt.show()
    ##################################


    ##save training model
    torch.save(net, StorageLocation + 'Exp_'+str(FLAG) + '_r' + repr(count) +'.pkl')

    print('第{}次实验，模型训练阶段完成！'.format(count))




    ######################################### Testing ####################################
    y_disp = np.zeros([All_data.shape[0]])
    y_disp[trn_num] = y_trn

    y_pred_tes = y_disp.copy()

    start = 0

    end = np.min([start + 100, len(tes_num)])

    ###################################### 分块测试 #######################################
    part_num = int(len(tes_num) / 100) + 1
    print('需要分成{}块来测试'.format(part_num))

    tsne_tes = []
    tsne_tes_y = []

    tes_time1 = time.time()

    for i in range(0, part_num):
        tsne_tes_part = []
        tsne_tes_y_part = []
        ###################### label ######################
        tes_num_part = tes_num[start:end]

        y_tes = All_data[tes_num_part, -1]  # label
        tes_YY = torch.from_numpy(y_tes - 1)  # (100,)

        ######################  data ######################
        a = product(c, FLAG)

        tes_spe, tes_spat, tes_num_part = a.production_data_valtespre_SpeAll(tes_num_part, half_s, image_3d_mat_origin, All_data_Spe_mat_origin, flag='Tes')

        tes_spat = tes_spat[:, :, :, :]
        tes_spe = tes_spe[:, :, np.newaxis]

        tes_XX_spat = torch.from_numpy(tes_spat.transpose(0, 3, 1, 2))
        tes_XX_spe = torch.from_numpy(tes_spe.transpose(0, 2, 1))

        del tes_spat, tes_spe

        ######### 推断，预测集 #########
        tes_dataset = TensorDataset(tes_XX_spe, tes_XX_spat, tes_YY)
        tes_loader = DataLoader(tes_dataset, batch_size=batchsize)

        net = torch.load(StorageLocation + 'Exp_' + str(FLAG) + '_r' + repr(count) + '.pkl', map_location='cpu')

        net = net.cuda()
        torch.set_num_threads(1)
        a = operate()

        y_pred_tes_part, tsne_tes_part, tsne_tes_y_part = a.inference(tsne_tes_part, tsne_tes_y_part, net, tes_loader, criterion, FLAG='TEST')


        y_pred_tes[tes_num_part] = y_pred_tes_part

        start = end
        end = np.min([start + 100, len(tes_num)])

        tsne_tes.append(tsne_tes_part)
        tsne_tes_y.append(tsne_tes_y_part)

    tes_time2 = time.time()
    #########################

    ####################################### Assess, 测试集 ###########################################
    print('==================Test set=====================')
    y_tes = All_data[tes_num, -1]  # label
    tes_YY = torch.from_numpy(y_tes - 1)
    y_tes = tes_YY.numpy() + 1
    print('第{}次实验，测试集OA={}'.format(count,np.mean(y_tes==y_pred_tes[tes_num])))  # corresponding location of 'test' pixels
    print('第{}次实验，测试集Kappa={}'.format(count,cohen_kappa_score(y_tes,y_pred_tes[tes_num])))
    runtime2 = time.time()

    ########## 各类别精度
    num_tes=np.zeros([categories-1])
    num_tes_pred=np.zeros([categories-1])
    for k in y_tes:
        num_tes[k-1]=num_tes[k-1]+1
    for j in range(y_tes.shape[0]):
        if y_tes[j]==y_pred_tes[tes_num][j]:
            num_tes_pred[y_tes[j]-1]=num_tes_pred[y_tes[j]-1]+1

    Acc=num_tes_pred/num_tes*100

    Experiment_result[0, count] = np.mean(y_tes == y_pred_tes[tes_num]) * 100  # OA
    Experiment_result[1, count] = np.mean(Acc)  # AA
    Experiment_result[2, count] = cohen_kappa_score(y_tes, y_pred_tes[tes_num]) * 100  # Kappa

    Experiment_result[3, count] = trn_time2 - trn_time1
    Experiment_result[4, count] = tes_time2 - tes_time1
    Experiment_result[5, count] = runtime2 - runtime1

    Experiment_result[6:, count] = Acc

    ############################ Output Excel #############################
    # 准备数据
    data_df = pd.DataFrame(Experiment_result)
    writer = pd.ExcelWriter(StorageLocation + str(FLAG) + '_' + str(int(Experiment_result[0, 0]*100)) + '.xls')
    data_df.to_excel(writer, 'Acc&Time')
    writer.save()
    #######################################################################
    print('第{}次实验，模型评估阶段完成！'.format(count))



######### 计算多次实验的均值与标准差并保存 #############
Experiment_result[:,-2]=np.mean(Experiment_result[:,0:-2],axis=1)
Experiment_result[:,-1]=np.std(Experiment_result[:,0:-2],axis=1)
scio.savemat(StorageLocation + str(FLAG) + '_AllResults' + repr(int(Experiment_result[0, -2]*100)) + '.mat',{'data':Experiment_result})
######### Output Mean&Std to Excel #######
data_df = pd.DataFrame(Experiment_result)
writer = pd.ExcelWriter(
    StorageLocation + str(FLAG) + '_AllMeanStd_' + repr(int(Experiment_result[0, -2]*100)) + '.xls')
data_df.to_excel(writer, 'All&Mean&Std')
writer.save()
#######################################

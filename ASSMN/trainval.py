import os
import argparse
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as scio
import torch
import torch.nn as nn
import cv2
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import cohen_kappa_score
#from thop import profile
from func import load,product,preprocess
from model import ASSMN, operate

## GPU_configration

# USE_GPU=True
# if USE_GPU:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# else:
#     device=torch.device('cpu')
#     print('using device:',device)

############ parameters setting ############

parser = argparse.ArgumentParser(description="ASSMN Training")
## dataset setting
parser.add_argument('--dataset', type=str, default='ksc',
                    choices=['indian','pavia','ksc'],
                    help='dataset name (default: indian)')
## dimension reduction setting
parser.add_argument('--dr-num', type=int, default=4,
                    help='dimension reduction components number')
parser.add_argument('--dr-method', type=str, default='pca',
                    choices=['pca','ica'],
                    help='dimension reduction way (default: pca)')
## normalization setting
parser.add_argument('--mi', type=int, default=-1,
                    help='min normalization range')
parser.add_argument('--ma', type=int, default=1,
                    help='max normalization range')
## data preparation setting
parser.add_argument('--half-size', type=int, default=13,
                    help='half of patch size')
parser.add_argument('--rsz', type=int, default=27,
                    help='resample size')
## experimental setting
parser.add_argument('--experiment-num', type=int, default=1,
                    help='experiment trials number')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--weight-decay', type=float, default=1e-5,
                    help='weight decay')
## model setting (SeMN)
parser.add_argument('--scheme', type=int, default=1,
                    help='schemes in SeMN (1 or 2)')
parser.add_argument('--strategy', type=str, default='s2',
                    choices=['s1', 's2'],help='choose strategy in scheme2 of SeMN')
parser.add_argument('--spec-time-steps', type=int, default=2,
                    help='time steps in scheme2 of SeMN with strategy s2')
## model setting (SaMN)
parser.add_argument('--group', type=str, default='alternate',
                    choices=['traditional','alternate'],
                    help='group strategy in SaMN')
parser.add_argument('--seq', type=str, default='cascade',
                    choices=['plain','cascade'],
                    help='input sequence styles in SaMN')
## network implementation setting
parser.add_argument('--oly-se', action='store_true', default=False,
                    help='only implement SeMN')
parser.add_argument('--oly-sa', action='store_true', default=False,
                    help='only implement SaMN')
parser.add_argument('--npi-num', type=int, default=2,
                    help='npi number, choose[0-9]')

args = parser.parse_args()

drop_p=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
num_patch_inrow=[1,2,3,4,5,6,7,8,9,10]

ii=args.npi_num
jj=8


wsz=[args.rsz//num_patch_inrow[ii],args.rsz//num_patch_inrow[ii],args.rsz//num_patch_inrow[ii]]
strd=[args.rsz//num_patch_inrow[ii],args.rsz//num_patch_inrow[ii],args.rsz//num_patch_inrow[ii]]


if args.oly_se and not args.oly_sa:
    print('Implement Spectral Network!')
elif not args.oly_se and args.oly_sa:
    print('Implement Spatial Network!')
elif not args.oly_se and not args.oly_sa:
    print('Implement Spectral & Spatial Joint Network!')
else:
    raise NotImplementedError

############# load dataset(indian_pines & pavia_univ...)######################

a=load()

All_data,labeled_data,rows_num,categories,r,c,FLAG=a.load_data(flag=args.dataset)

print('Data has been loaded successfully!')

##################### Demision reduction & normlization ######################

a=preprocess(args.dr_method,args.dr_num) #PCA & ICA

Alldata_DR=a.Dim_reduction(All_data)

print('Dimension reduction successfully!')

a = product(c, FLAG)

All_data_norm=a.normlization(All_data[:,1:-1],args.mi,args.ma)#spec

Alldata_DR=a.normlization(Alldata_DR,args.mi,args.ma)#spat

image_data3D_DR=Alldata_DR.reshape(r,c,-1)

print('Image normlization successfully!')

########################### Data preparation ##################################

image_3ddr_lr=np.fliplr(image_data3D_DR)
image_3ddr_ud=np.flipud(image_data3D_DR)
image_3ddr_corner=np.fliplr(np.flipud(image_data3D_DR))

image_3ddr_temp1=np.hstack((image_3ddr_corner,image_3ddr_ud,image_3ddr_corner))
image_3ddr_temp2=np.hstack((image_3ddr_lr,image_data3D_DR,image_3ddr_lr))
image_3ddr_merge=np.vstack((image_3ddr_temp1,image_3ddr_temp2,image_3ddr_temp1))

image_3ddr_mat_origin=image_3ddr_merge[(r-args.half_size):2*r+args.half_size,(c-args.half_size):2*c+args.half_size,:]

print(image_3ddr_mat_origin.shape)

# plt.imshow(image_3ddr_mat_origin[:,:,0])
# plt.show()

print('image edge enhanced Finished!')

del image_3ddr_lr,image_3ddr_ud,image_3ddr_corner,image_3ddr_temp1,image_3ddr_temp2,image_3ddr_merge

##################################### trn/val ####################################

#Experimental memory
Experiment_result=np.zeros([categories+4,12])#OA,AA,kappa,trn_time,tes_time

#kappa
kappa=0

for count in range(0,args.experiment_num):

    a = product(c, FLAG)

    rows_num,trn_num,tes_num,pre_num=a.generation_num(labeled_data,rows_num,All_data)

    #################################### trn #####################################

    ########### spec data ##########

    trn_spec=All_data_norm[trn_num,:]
    trn_XX_spec=torch.from_numpy(trn_spec)

    ############ label #############

    y_trn = All_data[trn_num, -1]
    trn_YY = torch.from_numpy(y_trn - 1)  # label start from 0

    ############ spat data #########

    a = product(c, FLAG)

    trn_spat,trn_num,_=a.production_data_trn(rows_num,trn_num,args.half_size,image_3ddr_mat_origin)
    trn_spat = a.resample(trn_spat,args.rsz)
    trn_XX_spat = torch.from_numpy(trn_spat.transpose(0, 3, 1, 2))  # (B,C,H,W)

    del trn_spat

    print('Experiment {}，Training spatial dataset preparation Finished!'.format(count))

    ########## training #############

    torch.cuda.empty_cache()#GPU memory released

    trn_dataset=TensorDataset(trn_XX_spec,trn_XX_spat,trn_YY)
    trn_loader=DataLoader(trn_dataset,batch_size=args.batch_size,sampler=SubsetRandomSampler(range(trn_XX_spat.shape[0])))

    net = ASSMN(trn_XX_spec.shape[1],trn_XX_spat.shape[1],args.scheme,args.strategy,
               args.spec_time_steps,wsz,strd,categories-1,num_patch_inrow[ii],drop_p[jj],
               args.group,args.seq,args.oly_se,args.oly_sa)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    if args.oly_se and not args.oly_sa:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=args.weight_decay)
    elif args.oly_sa and not args.oly_se:
        optimizer = torch.optim.SGD(net.parameters(),momentum=0.9, lr=args.lr,
                                    weight_decay=args.weight_decay)
    elif not args.oly_se and not args.oly_sa:
        optimizer = torch.optim.SGD(net.parameters(), momentum=0.9, lr=args.lr,
                                    weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120,150,180], gamma=0.5)

    loss_trn = []

    trn_time1 = time.time()

    for i in range(1, args.epochs):
        b=operate()
        loss_trn = b.train(i, loss_trn, net, optimizer, scheduler, trn_loader, criterion)

    trn_time2 = time.time()

    # plt.figure(1)
    # plt.plot(np.array(loss_trn), label='Training')
    # plt.legend()
    # plt.show()

    print('########### Experiment {}，Model Training Period Finished! ############'.format(count))

    #################################### val ####################################

    ########## spec data ###########

    tes_spec=All_data_norm[tes_num,:]
    tes_XX_spec=torch.from_numpy(tes_spec)

    ############ label #############

    y_tes = All_data[tes_num, -1]#label
    tes_YY = torch.from_numpy(y_tes - 1)

    ############ spat data #########

    a = product(c, FLAG)

    tes_spat, tes_num = a.production_data_valtespre(tes_num, args.half_size, image_3ddr_mat_origin, flag='Tes')
    tes_spat = a.resample(tes_spat, args.rsz)
    tes_XX_spat = torch.from_numpy(tes_spat.transpose(0, 3, 1, 2))

    del tes_spat

    print('Experiment {}，Testing spatial dataset preparation Finished!'.format(count))

    ################### testing ################

    tes_dataset=TensorDataset(tes_XX_spec,tes_XX_spat,tes_YY)
    tes_loader=DataLoader(tes_dataset,batch_size=100)

    net=net.cuda()

    a=operate()

    tes_time1 = time.time()
    y_pred_tes=a.inference(net,tes_loader,criterion,FLAG='TEST')
    tes_time2 = time.time()

    ####################################### assess ###########################################

    print('==================Test set=====================')
    y_tes = tes_YY.numpy() + 1
    print('Experiment {}，Testing set OA={}'.format(count,np.mean(y_tes==y_pred_tes)))
    print('Experiment {}，Testing set Kappa={}'.format(count,cohen_kappa_score(y_tes,y_pred_tes)))

    flag1='se' if args.oly_se else 'nose'
    flag2='sa' if args.oly_sa else 'nosa'

    if cohen_kappa_score(y_tes,y_pred_tes)>=kappa:
        torch.save(net,'assmn_'+str(FLAG)+'_scheme_'
                   +str(args.scheme)+'_'+str(args.strategy)+'_'+str(flag1)+'_'+str(flag2)+'.pkl')
        kappa=cohen_kappa_score(y_tes,y_pred_tes)

    ## Detailed information (every classes accuracy)

    num_tes=np.zeros([categories-1])
    num_tes_pred=np.zeros([categories-1])
    for k in y_tes:
        num_tes[k-1]=num_tes[k-1]+1
    for j in range(y_tes.shape[0]):
        if y_tes[j]==y_pred_tes[j]:
            num_tes_pred[y_tes[j]-1]=num_tes_pred[y_tes[j]-1]+1

    Acc=num_tes_pred/num_tes*100

    Experiment_result[0,count]=np.mean(y_tes==y_pred_tes)*100#OA
    Experiment_result[1,count]=np.mean(Acc)#AA
    Experiment_result[2,count]=cohen_kappa_score(y_tes,y_pred_tes)*100#Kappa
    Experiment_result[3, count] = trn_time2 - trn_time1
    Experiment_result[4, count] = tes_time2 - tes_time1
    Experiment_result[5:,count]=Acc

    print('########### Experiment {}，Model assessment finished！！! ###########'.format(count))

########## mean value & standard deviation #############

Experiment_result[:,-2]=np.mean(Experiment_result[:,0:-2],axis=1)
Experiment_result[:,-1]=np.std(Experiment_result[:,0:-2],axis=1)

scio.savemat('assmn_result_'+str(FLAG)+'.mat',{'data':Experiment_result})

np.save('trn_num_'+str(FLAG)+'.npy',trn_num)
np.save('pre_num_'+str(FLAG)+'.npy',pre_num)
np.save('y_trn_'+str(FLAG)+'.npy',y_trn)
np.save('image_3d_mat_origin_'+str(FLAG)+'.npy',image_3ddr_mat_origin)

############# display ##############

y_disp=np.zeros([All_data.shape[0]])

y_disp[trn_num]=y_trn
y_disp[tes_num]=y_pred_tes

y_disp_gt=y_disp.copy()
y_disp_gt[tes_num]=y_tes

# plt.subplots(figsize=[10,10])
# ax1=plt.subplot(1,2,1)
# plt.xlabel('TEST')
# a1=plt.imshow(y_disp.reshape(r,c),cmap='jet')
# plt.xticks([])
# plt.yticks([])
#
# ax2=plt.subplot(1,2,2)
# plt.xlabel('gt')
# a2=plt.imshow(y_disp_gt.reshape(r,c),cmap='jet')
# plt.xticks([])
# plt.yticks([])
#
# plt.show()

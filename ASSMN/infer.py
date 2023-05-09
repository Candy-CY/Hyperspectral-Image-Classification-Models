import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import cv2
import argparse
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from func import load,product,preprocess
from model import ASSMN, operate

## GPU_configration

# USE_GPU=True
# if USE_GPU:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# else:
#     device=torch.device('cpu')

############ parameters setting ############

parser = argparse.ArgumentParser(description="ASSMN Testing")

parser.add_argument('--dataset', type=str, default='indian',
                    choices=['indian','pavia','ksc','houston'],
                    help='dataset name (default: indian)')
## spectral normalization setting （spatial part don't need to normal for load normalized matrix）
parser.add_argument('--mi', type=int, default=-1,
                    help='min normalization range')
parser.add_argument('--ma', type=int, default=1,
                    help='max normalization range')
## data preparation setting
parser.add_argument('--half-size', type=int, default=13,
                    help='half of patch size')
parser.add_argument('--rsz', type=int, default=27,
                    help='resample size')
## sample number of every chunk
parser.add_argument('--bz', type=int, default=50000,
                    help='half of patch size')
## model setting (SeMN)
parser.add_argument('--scheme', type=int, default=1,
                    help='schemes in SeMN (1 or 2)')
parser.add_argument('--strategy', type=str, default='s2',
                    choices=['s1', 's2'],help='choose strategy in scheme2 of SeMN')
## network implementation setting
parser.add_argument('--oly-se', action='store_true', default=False,
                    help='only implement SeMN')
parser.add_argument('--oly-sa', action='store_true', default=False,
                    help='only implement SaMN')

args = parser.parse_args()

## load data

a=load()

All_data, _, _, _,r,c,FLAG=a.load_data(flag=args.dataset)

a = product(c, FLAG)

All_data_norm=a.normlization(All_data[:,1:-1],args.mi,args.ma)#spec

trn_num=np.load('trn_num_'+str(FLAG)+'.npy')
pre_num=np.load('pre_num_'+str(FLAG)+'.npy')
y_trn=np.load('y_trn_'+str(FLAG)+'.npy')
image_3d_mat_origin=np.load('image_3d_mat_origin_'+str(FLAG)+'.npy')

if args.oly_se and not args.oly_sa:
    print('Implement Spectral Network!')
elif not args.oly_se and args.oly_sa:
    print('Implement Spatial Network!')
elif not args.oly_se and not args.oly_sa:
    print('Implement Spectral & Spatial Joint Network!')
else:
    raise NotImplementedError

flag1='se' if args.oly_se else 'nose'
flag2='sa' if args.oly_sa else 'nosa'

net=torch.load('assmn_'+str(FLAG)+'_scheme_'+str(args.scheme)+'_'+str(args.strategy)+'_'
               +str(flag1)+'_'+str(flag2)+'.pkl',map_location='cpu')

net=net.cuda()

criterion = torch.nn.CrossEntropyLoss()

### label config
y_disp=np.zeros([All_data.shape[0]])
y_disp[trn_num]=y_trn

y_disp_all=y_disp.copy()

start=0
end=np.min([start+args.bz,pre_num.shape[0]])

###################################### 预测 #######################################

part_num=int(pre_num.shape[0]/args.bz)+1

print('Need to split {} parts for prediction'.format(part_num))

for i in range(0,part_num):

    ###################### label ######################

    pre_num_part=pre_num[start:end]

    y_pre=All_data[pre_num_part,-1]#include background

    pre_YY = torch.from_numpy(np.ones([y_pre.shape[0]]))

    ######################  data ######################

    #spec
    pre_spec = All_data_norm[pre_num_part, :]
    pre_XX_spec = torch.from_numpy(pre_spec)

    #spat
    a = product(c, FLAG)

    pre_spat, pre_num_part = a.production_data_valtespre(pre_num_part, args.half_size, image_3d_mat_origin, flag='Pre')
    pre_spat = a.resample(pre_spat, args.rsz)
    pre_XX_spat=torch.from_numpy(pre_spat.transpose(0, 3, 1, 2))

    del pre_spat

    ######### inferring #########

    pre_dataset=TensorDataset(pre_XX_spec, pre_XX_spat,pre_YY)
    pre_loader=DataLoader(pre_dataset,batch_size=500)

    a=operate()
    y_pred_pre=a.inference(net,pre_loader,criterion,FLAG='PRED')

    print('Part {} Prediction Finished！！！'.format(i))

    y_disp_all[pre_num_part]=y_pred_pre

    start=end
    end=np.min([start+args.bz,pre_num.shape[0]])

###################### display ##########################

# plt.xlabel('All image')
# plt.imshow(y_disp_all.reshape(r,c),cmap='jet')
# plt.xticks([])
# plt.yticks([])
#
# plt.show()

################# save whole prediction map #############

# cv2.imwrite('assmn_all_'+str(FLAG)+'_scheme_'+str(args.scheme)+'_'+str(args.strategy)+'_'
#                +str(flag1)+'_'+str(flag2)+'.png', y_disp_all.reshape(r,c))

plt.subplots(figsize=[10,10])
a1=plt.imshow(y_disp_all.reshape(r,c),cmap='jet')
plt.xticks([])
plt.yticks([])
plt.savefig('assmn_all_'+str(FLAG)+'.png',dpi=600,bbox_inches='tight')

print('######## Result displaying & saving finished！！#########')
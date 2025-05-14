import cv2
import torch
from utils.compute_metrics import compute_sam,compute_psnr,compute_ergas,compute_cc,compute_rmse
import numpy as np
import scipy.io as sio
from config import args
from Data_loader import Dataset
import spectral as spy
import matplotlib.pyplot as plt

train_dataset=Dataset(args)
hsi_channels=train_dataset.hsi_channel
msi_channels=train_dataset.msi_channel

from arch.spectral_upsample import Spectral_upsample

Spectral_up_net   = Spectral_upsample(args,msi_channels,hsi_channels,init_type='normal', init_gain=0.02,initializer=False)
Spectral_up_net.load_state_dict(torch.load(r'checkpoints/houston18_5_S1=0.0008_40000_10000_S2=0.0013_200000_20000/spectral_upsample.pth',map_location=args.device))

lhsi=train_dataset[0]["lhsi"].unsqueeze(0).to(args.device)# unsqueeze添加batch维度，将C,H,W变为B,C,H,W 满足pytorch输入要求
hmsi=train_dataset[0]['hmsi'].unsqueeze(0).to(args.device)
hhsi=train_dataset[0]['hhsi'].unsqueeze(0).to(args.device)
lrmsi_frommsi=train_dataset[0]['lrmsi_frommsi'].unsqueeze(0).to(args.device)
lrmsi_fromlrhsi=train_dataset[0]['lrmsi_fromlrhsi'].unsqueeze(0).to(args.device)

hhsi_true=hhsi.detach().cpu().numpy()[0].transpose(1,2,0)
out_lrhsi_true=lhsi.detach().cpu().numpy()[0].transpose(1,2,0)
out_msi_true=hmsi.detach().cpu().numpy()[0].transpose(1,2,0)
out_frommsi_true= lrmsi_frommsi.detach().cpu().numpy()[0].transpose(1,2,0)
out_fromlrhsi_true= lrmsi_fromlrhsi.detach().cpu().numpy()[0].transpose(1,2,0)

'''
生成predicted X
'''
est_hhsi=Spectral_up_net(hmsi).detach().cpu().numpy()[0].transpose(1,2,0) #对msi上采样到hhsi

test_message_specUp='生成hhsi 与目标hhsi \ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}, dataname{}'.\
                          format(
                                 np.mean( np.abs( hhsi_true-est_hhsi ) ) ,
                                 compute_sam(hhsi_true,est_hhsi) ,
                                 compute_psnr(hhsi_true,est_hhsi) ,
                                 compute_ergas(hhsi_true,est_hhsi,args.scale_factor) ,
                                 compute_cc(hhsi_true,est_hhsi),
                                 compute_rmse(hhsi_true,est_hhsi),
                                 args.data_name
                                 )

print(test_message_specUp)

sio.savemat(r'checkpoints\Final.mat',{args.data_name+'_final':est_hhsi})
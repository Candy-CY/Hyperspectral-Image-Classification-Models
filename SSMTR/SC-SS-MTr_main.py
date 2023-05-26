# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 19:13:08 2022

@author: HLB
"""

import record

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import time
import datetime
import numpy as np


from config import load_args
from data_read import readdata
from generate_pic import generate
from hyper_dataset import HyperData
from augment import CenterResizeCrop
from util_CNN import test_batch, pre_train
from pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from HSI_ConSupMaeModel import mae_vit_HSI_patch3, vit_HSI_patch3

args = load_args()

mask_ratio = args.mask_ratio
windowsize = args.windowsize
dataset = args.dataset
type = args.type
num_epoch = args.epochs
lr = args.lr
train_num_per = args.train_num_perclass
num_of_ex = 10
batch_size= args.batch_size
net_name = args.cl_mode+'_SC_SS_MTr' 

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')
halfsize = int((windowsize-1)/2)
_, _, _, _,_, _, _, _, _,_, gt,s = readdata(type, dataset, windowsize,train_num_per, 1000, 0)
num_of_samples = int(s * 0.2)
nclass = np.max(gt)
print(nclass)


KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((num_of_ex, nclass))
af_result = np.zeros([nclass+3, num_of_ex])
criterion = nn.CrossEntropyLoss()

for num in range(0,num_of_ex):
    print('num:', num)    
    train_image, train_label, validation_image1, validation_label1, nTrain_perClass, nvalid_perClass, train_index, val_index, index,image, gt,s = readdata(type, dataset, windowsize,train_num_per,num_of_samples,num)
    ind = np.random.choice(validation_image1.shape[0], 100, replace = False)
    validation_image = validation_image1[ind]
    validation_label= validation_label1[ind]
    nvalid_perClass = np.zeros_like(nvalid_perClass)
    nband = train_image.shape[3]


    train_num = train_image.shape[0] 
    train_image = np.transpose(train_image,(0,3,1,2))

    validation_image = np.transpose(validation_image,(0,3,1,2))
    validation_image1 = np.transpose(validation_image1,(0,3,1,2))

    if args.augment:
        transform_train = [CenterResizeCrop(scale_begin = args.scale, windowsize = windowsize)]
        train_dataset = HyperData((train_image, train_label), transform_train)
    else:
        train_dataset = TensorDataset(torch.tensor(train_image), torch.tensor(train_label))    
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

    print("=> creating model '{}'".format(net_name))
    

    ########################   maevit的训练
    net = mae_vit_HSI_patch3(img_size=(27,27), in_chans=nband, hid_chans = args.hid_chans, embed_dim=args.encoder_dim, depth=args.encoder_depth, num_heads=args.encoder_num_heads,  mlp_ratio=args.mlp_ratio,
                              decoder_embed_dim=args.decoder_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads, nb_classes=nclass)
    net.cuda()
    optimizer = optim.Adam(net.parameters(),lr = lr, weight_decay= 1e-4) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,  T_0=5,T_mult=2)



    
    tic1 = time.time()
    for epoch in range(num_epoch):
        net.train()
        total_loss = 0
        for idx, (x,y) in enumerate(train_loader):                        

            x = x.cuda()
            y = y.cuda()
            rec_loss, cl_loss, _, _, logits = net(x, y,mask_ratio=mask_ratio, mode = args.cl_mode, temp = args.cl_temperature)           
            cls_loss = criterion(logits / args.temperature, y) * args.cls_loss_ratio
            loss = cls_loss + rec_loss + cl_loss * args.cl_loss_ratio
            
            optimizer.zero_grad()                             
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss
            
        scheduler.step()
        total_loss = total_loss/(idx+1)
        state = {'model':net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        
        print('epoch:',epoch,'loss:',total_loss.data.cpu().numpy())
    toc1 = time.time()                
    torch.save(state, './net.pt')
    
    # ########################   vit的finetune 
    model = vit_HSI_patch3(img_size=(windowsize,windowsize), in_chans=nband, hid_chans = args.hid_chans, embed_dim=args.encoder_dim, depth=args.encoder_depth, num_heads=args.encoder_num_heads,  mlp_ratio=args.mlp_ratio,num_classes = nclass, global_pool=False).cuda()
    checkpoint = torch.load('./net.pt')
    checkpoint_model = checkpoint['model']  
    
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
            
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
    
    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)   
    tic2 = time.time()
  
    optimizer = optim.Adam(model.parameters(),lr = lr, weight_decay= 1e-4) 

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [80, 140], gamma = 0.1, last_epoch=-1)
    model = pre_train(model, train_image, train_label,validation_image, validation_label, 180, optimizer, scheduler, batch_size, val = False)
    toc2 = time.time()
    
    true_cla, overall_accuracy, average_accuracy, kappa, true_label, test_pred, test_index, cm, pred_array = test_batch(model.eval(), image, index, 100,  nTrain_perClass,nvalid_perClass, halfsize)
    toc3 = time.time()
    
    af_result[:nclass,num] = true_cla
    af_result[nclass,num] = overall_accuracy
    af_result[nclass+1,num] = average_accuracy
    af_result[nclass+2,num] = kappa    

        
    OA.append(overall_accuracy)
    AA.append(average_accuracy)
    KAPPA.append(kappa)
    TRAINING_TIME.append(toc1 - tic1 + toc2 - tic2)
    TESTING_TIME.append(toc3 - toc2)
    ELEMENT_ACC[num, :] = true_cla
    classification_map, gt_map = generate(image, gt, index, nTrain_perClass, nvalid_perClass, test_pred, overall_accuracy, halfsize, dataset, day_str, num, net_name)
jingdu = np.mean(af_result, axis = 1)
print("--------" + net_name + " Training Finished-----------")
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                      'records/' + dataset + '/'+ net_name + '_' + day_str+ '_' + str(args.epochs) + '_train_num：' + str(train_image.shape[0]) +'_windowsize：' + str(windowsize)+'_mask_ratio_' + str(mask_ratio) + '_temperature_' + str(args.temperature) +
                      '_augment_' + str(args.augment) +'_aug_scale_' + str(args.scale) + '_CLloss_ratio_' + str(args.cl_loss_ratio) + '_cl_temp_' + str(args.cl_temperature) +'_loss_ratio_' + str(args.cls_loss_ratio)+'_decoder_dim_' + str(args.decoder_dim) + '_decoder_depth_' + str(args.decoder_depth)+ '.txt') 




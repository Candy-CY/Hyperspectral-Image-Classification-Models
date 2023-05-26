# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:08:36 2022

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
from util_CNN import test_batch, tr_acc
from HSI_SupMaeModel import vit_HSI_patch3
from timm.models.layers import trunc_normal_


args = load_args()
windowsize = args.windowsize
dataset = args.dataset
num_of_samples = 1000
type = args.type
num_epoch = 180
lr = args.lr
train_num_per = args.train_num_perclass
num_of_ex = 1
net_name = 'vit_HSI' 



day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')
halfsize = int((windowsize-1)/2)
_, _, _, _,_, _, _, _, _,_, gt,_ = readdata(type, dataset, windowsize,train_num_per,num_of_samples, 0)
nclass = np.max(gt)
print(nclass)
batch_size= args.batch_size
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
    train_image, train_label, validation_image1, validation_label1,nTrain_perClass, nvalid_perClass, train_index, val_index, index,image, gt,s = readdata(type, dataset, windowsize,train_num_per,num_of_samples, num)
    ind = np.random.choice(validation_image1.shape[0], 200, replace = False)
    validation_image = validation_image1[ind]
    validation_label= validation_label1[ind]
    nvalid_perClass = np.zeros_like(nvalid_perClass)
    nband = train_image.shape[3]


    train_num = train_image.shape[0] 
    train_image = np.transpose(train_image,(0,3,1,2))

    validation_image = np.transpose(validation_image,(0,3,1,2))

    print("=> creating model '{}'".format(net_name))
    
    ########################基础vit的训练与测试
    model = vit_HSI_patch3(img_size=(27,27), in_chans=nband, hid_chans = args.hid_chans, embed_dim=args.encoder_dim, depth=args.encoder_depth, num_heads=args.encoder_num_heads,  mlp_ratio=args.mlp_ratio,num_classes = nclass).cuda()
    trunc_normal_(model.head.weight, std=2e-5) 

    optimizer = optim.Adam(model.parameters(),lr = lr,weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [80, 140], gamma = 0.1, last_epoch=-1)
    tic1 = time.time() 
    

    train_dataset = TensorDataset(torch.tensor(train_image), torch.tensor(train_label)) 
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    
    for i in range(num_epoch): 
        model.train()
        train_loss = 0
        for idx, (image_batch, label_batch) in enumerate(train_loader):        

            x = image_batch.cuda()
            label_batch = label_batch.cuda()    
            logits = model(x)
            loss = criterion(logits, label_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + loss
        scheduler.step()
        train_loss = train_loss / (idx + 1)
        train_acc, train_loss = tr_acc(model.eval(), train_image, train_label)
        val_acc, val_loss = tr_acc(model.eval(), validation_image, validation_label)
        print("epoch {}, training loss: {:.4f}, train acc:{:.4f}, valid acc:{:.4f}".format(i, train_loss.item(), train_acc*100, val_acc*100))
    toc1 = time.time() 
    
    
    
    tic2 = time.time() 
    true_cla, overall_accuracy, average_accuracy, kappa, true_label, test_pred, test_index, af_cm, pred_array = test_batch(model.eval(), image, index, 100,  nTrain_perClass,nvalid_perClass, halfsize)
    toc2 = time.time()
    
    af_result[:nclass,num] = true_cla
    af_result[nclass,num] = overall_accuracy
    af_result[nclass+1,num] = average_accuracy
    af_result[nclass+2,num] = kappa  
        
    OA.append(overall_accuracy)
    AA.append(average_accuracy)
    KAPPA.append(kappa)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[num, :] = true_cla
    classification_map, gt_map = generate(image, gt, index, nTrain_perClass, nvalid_perClass, test_pred, overall_accuracy, halfsize, dataset, day_str, num, net_name)
print("--------" + net_name + " Training Finished-----------")
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                      'records/' + dataset + '/'+ net_name + '_' + day_str + '_' + dataset +'_augment_' +str(args.augment) + '_aug_scale_' + str(args.scale)+ '_split：' + str(train_image.shape[0]) +'windowsize：' + str(windowsize)+ '_dim_' + str(args.encoder_dim) + '_depth_' + str(args.encoder_depth)+ '_ratio_' + str(args.mlp_ratio)+ '_heads_' + str(args.encoder_num_heads)+ '_.txt') 
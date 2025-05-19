# -*- coding: utf-8 -*-
import os
import argparse
import libmr
import rscls
import glob
import time
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# import ipdb

from scipy import io
from scipy.io import loadmat
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics.pairwise import paired_distances as dist
from torch.optim.lr_scheduler import LambdaLR   
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

from Hyper import imgDraw
from utils import seed_torch
from utils import to_categorical
# from utils import pca   
from HypertorchProser import ResNet999
from My_Dataset import CustomDataset


os.environ['CUDA_VISIBLE_DEVICES']='0'
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--numTrain', type=int, default=20)
parser.add_argument('--data', type=int, default=3)   
parser.add_argument('--Run_times', type=int, default=10)    
parser.add_argument('--batch_size', type=int, default=32)   
parser.add_argument('--output', type=str, default='output/')
parser.add_argument('--lr1', type=float, default=1)
parser.add_argument('--lr2', type=float, default=0.1)
parser.add_argument('--dummy_num', type=int, default=2)
parser.add_argument('--lamda', type=float, default=0.6)
parser.add_argument('--PPP', type=float, default=0.1)
parser.add_argument('--P1', type=float, default=0.1)
parser.add_argument('--P2', type=float, default=0.1)
parser.add_argument('--numproto', type=int, default=3)
parser.add_argument('--proser_epoch', type=int, default=50)
parser.add_argument('--proser_batchsize', type=int, default=256)
parser.add_argument('--latentdim', type=int, default=25)
parser.add_argument('--num_epochs1', type=int, default=170)
parser.add_argument('--num_epochs2', type=int, default=130)
parser.add_argument('--Sed', type=int, default=1)
parser.add_argument('--lr_proser', type=float, default=0.001)
parser.add_argument('--YuZhi', type=float, default=0.5)
args = parser.parse_args()  



dataset = 'data/Muufl/muufl_im.mat'
gt = 'data/Muufl/muufl_raw_gt.mat'
num_epochs1 = 1100
num_epochs2 = 1000
proser_batchsize = args.proser_batchsize
proser_epoch = 80
dummy_num = args.dummy_num
latentdim = 15
P1 = 1
lamda = 0.4
numproto = 1
PPP = 0.1
seed = [0,1,3,6,8]




key = dataset.split('/')[-1].split('_')[0]
spath = args.output + key + '_' + str(args.numTrain) + '/'
os.makedirs(spath, exist_ok=True)


def lr_lambda(epoch):
    return args.lr1 if epoch < num_epochs1 else args.lr2


if args.data == 3: # 
    hsi = loadmat(dataset)['combinedData']
    gt = loadmat(gt)['label']
else:
    hsi = np.load(dataset).astype('float32')
    gt = np.load(gt).astype('int')

num_classes = np.max(gt)
row, col, layers = hsi.shape




acc = np.zeros([len(seed), 1])
OSR = np.zeros([len(seed), 1])
A = np.zeros([len(seed), num_classes])
k = np.zeros([len(seed), 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None
train_time = np.zeros([len(seed), 1])
test_time = np.zeros([len(seed), 1])




for Running in range(len(seed)):
    seed_torch(seed[Running])
    c1 = rscls.rscls(hsi, gt, cls=num_classes)
    c1.padding(9)
    x1_train, y1_train = c1.train_sample(args.numTrain) 
    x1_train, y1_train = rscls.make_sample(x1_train, y1_train) 

    if args.data == 3:
        x1_lidar_train = x1_train[:, :, :, -2:]
        x1_train = x1_train[:, :, :, :-2]
    else:
        x1_lidar_train = x1_train[:, :, :, -1]
        x1_lidar_train = x1_lidar_train[:, :, :, np.newaxis]
        x1_train = x1_train[:, :, :, :-1]

    y1_train = to_categorical(y1_train, num_classes)
    
    x1_train = x1_train.transpose((0,3,1,2))
    x1_lidar_train = x1_lidar_train.transpose((0,3,1,2))

    train_dataset = CustomDataset(torch.tensor(x1_train), torch.tensor(x1_lidar_train),torch.tensor(y1_train))
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.L1Loss()

    np.set_printoptions(threshold=np.inf)   # SET PRINTING


    model = ResNet999(args.data, layers - 1, 1, dummy_num,num_classes,n_sub_prototypes = numproto,latent_dim=latentdim,l=1,temp_intra = P1,temp_inter = args.P2).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = LambdaLR(optimizer, lr_lambda)
    # START TRAINING
    start_time = time.time()
    model.train()                                
    for epoch in range(num_epochs1 + num_epochs2):
        for x_HSI, x_lidar, y in train_data_loader:
            x_HSI = torch.FloatTensor(x_HSI).to(device)  
            x_lidar = torch.FloatTensor(x_lidar).to(device)  
            y = y.to(device).float()
            x = torch.concat([x_HSI,x_lidar],dim=1)

            optimizer.zero_grad()

            output1, output2, dummypre, latent_z, distt, kl_div ,output3 = model(x) 
            loss1 = criterion1(output1,torch.argmax(y,dim=1))
            loss2 = criterion2(output2,x_HSI)
            loss3 = criterion2(output3,x_lidar)

            lossProto  = model.GetlossMGPL(output1, output2, dummypre, latent_z, distt, kl_div, x, y)
            lossBefore = 0.5 * loss1 + 0.5 * loss2 + PPP * loss3
            
            loss4 = lamda * (lossProto['kld'] +  lossProto['ent']) + (1 - lamda) * lossProto['dis']
            loss = lossBefore + loss4


            
            loss.backward()
            optimizer.step()
        
        scheduler.step()  # Change learning_rate
        print(f"Epoch {epoch+1}, LossBefore: {lossBefore.item()},LossProto: {loss4.item()}")


    end_time = time.time()
    print('Training time:', end_time - start_time)
    train_time[Running] = end_time - start_time


    Proser_data_loader = DataLoader(train_dataset, batch_size=proser_batchsize, shuffle=True)
    start_time = time.time()
    
    model.fc_mu.weight.requires_grad = False
    model.fc_mu.bias.requires_grad = False
    model.fc_logvar.weight.requires_grad = False
    model.fc_logvar.bias.requires_grad = False
    model.prototypes.requires_grad = False
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr_proser,weight_decay=5e-4)
    
    for epoch in range(proser_epoch):
        train_loss = 0
        correct = 0
        total = 0
        for x, x_lidar, y in Proser_data_loader:
            
            x = torch.FloatTensor(x).to(device)  
            x_lidar = torch.FloatTensor(x_lidar).to(device)  
            x = torch.concat([x,x_lidar],dim=1)  
            y = y.to(device).float()
            y = torch.argmax(y,dim=1)
            optimizer.zero_grad()


            halflenth=int(len(x)/2)
            # beta = 0.5
            beta = torch.distributions.beta.Beta(1, 1).sample([]).item()   

            prehalfinputs = x[:halflenth]   
            prehalflabels = y[:halflenth]

            laterhalfinputs = x[halflenth:]
            laterhalflabels = y[halflenth:]

            index = torch.randperm(prehalfinputs.size(0)).cuda()   
            
            pre2embeddings = model.encoder(prehalfinputs)   
            mixed_embeddings = beta * pre2embeddings + (1 - beta) * pre2embeddings[index] 

            mix = model.latter(mixed_embeddings)
            prepre1,predummy,_,_ = model.fc(mix)
            prehalfoutput = torch.cat((prepre1,predummy),1)

            pre1, _, dummypre,_,_,_,_ = model(laterhalfinputs)
            latterhalfoutput = torch.cat((pre1,dummypre),1)   

            maxdummy, _ = torch.max(predummy.clone(),dim=1)  
            maxdummy = maxdummy.view(-1,1)  

            dummpyoutputs=torch.cat((pre1.clone(),maxdummy),dim=1)

            for i in range(len(dummpyoutputs)):
                # ipdb.set_trace()
                nowlabel = laterhalflabels[i]   
                dummpyoutputs[i][nowlabel] = -1e10   

            dummytargets = torch.ones_like(laterhalflabels) * num_classes 

            loss1 = criterion1(prehalfoutput, (torch.ones_like(prehalflabels)*num_classes).long().cuda())     
            loss2 = criterion1(latterhalfoutput,laterhalflabels )  
            loss3 = criterion1(dummpyoutputs, dummytargets)
            
            loss=0.001*loss1+loss2+loss3
            
            # loss = loss2 + loss3
            outputs=torch.cat((prehalfoutput,latterhalfoutput),0)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            accury = 100.*correct/total
        
        print(f"Epoch {epoch+1},Loss1: {loss1:.4f}, Loss2: {loss2:.4f}, Loss3: {loss3:.4f}")
        # print(f"Epoch {epoch+1}, Loss: {loss}, Accury: {accury}")

    end_time = time.time()
    print('Prosering time:', end_time - start_time)


    model.eval()  
    time3 = int(time.time())
    print('Start predicting...')

    model.eval()  
    pre_all = []
    pre_loss = []
    with torch.no_grad():  
        # for r in tqdm(range(row)):
        for r in range(row):

            row_samples = c1.all_sample_row(r)  
            
            row_samples_tensor = torch.tensor(row_samples, dtype=torch.float32)

            if args.data == 3:
                x_samples = row_samples_tensor[:, :, :, :-2]
                lidar_samples = row_samples_tensor[:, :, :, -2:]
            else:
                x_samples = row_samples_tensor[:, :, :, :-1]
                lidar_samples = row_samples_tensor[:, :, :, -1].unsqueeze(-1)
            
            # ipdb.set_trace()
            x_samples = x_samples.permute(0,3,1,2).to(device)
            lidar_samples = lidar_samples.permute(0,3,1,2).to(device)
            x = torch.cat((x_samples,lidar_samples),dim=1)
            pre_row, recons, _,_,_,_,recons2  = model(x)
            
            pre_all.append(pre_row.cpu().numpy())
 
            recons_loss = dist(recons.cpu().reshape(col, -1), x_samples.cpu().reshape(col, -1)) 
            pre_loss.append(recons_loss)

    pre_all = np.array(pre_all).astype('float64')
    pre_loss = np.array(pre_loss).reshape(-1).astype('float64')

        
    _, recons_train, _ ,_,_,_, recons_train2 = model(torch.concat([torch.FloatTensor(x1_train).to(device), torch.FloatTensor(x1_lidar_train).to(device)],dim=1))
    # train_loss = dist(recons_train.reshape(recons_train.shape[0], -1).cpu().detach().numpy(), 
    #                 x1_train.reshape(x1_train.shape[0], -1)) 
    if args.data == 3:
        train_loss = dist(recons_train.reshape(recons_train.shape[0], -1).cpu().detach().numpy(), 
                    x1_train.reshape(x1_train.shape[0], -1)) 
    else:
        train_loss = (dist(recons_train.reshape(recons_train.shape[0], -1).cpu().detach().numpy(), 
                    x1_train.reshape(x1_train.shape[0], -1)) + PPP * dist(recons_train2.reshape(recons_train2.shape[0], -1).cpu().detach().numpy(), 
                x1_lidar_train.reshape(x1_lidar_train.shape[0], -1)))


    time4 = int(time.time())
    print('predict time:',time4-time3)

    print('Start caculating open-set...')

    # ipdb.set_trace()


    mr = libmr.MR()
    mr.fit_high(train_loss, 20)
    wscore = mr.w_score_vector(pre_loss.astype(np.float64))
    mask = wscore > args.YuZhi 
    mask = mask.reshape(row, col)
    unknown = gt.max() + 1

    # for close set
    pre_closed = np.argmax(pre_all, axis=-1) + 1     
    imgDraw(pre_closed, spath + key + '_closed', path='./', show=False)

    # for open set
    pre_gsrl = deepcopy(pre_closed)
    pre_gsrl[mask == 1] = unknown 
    gt_new = deepcopy(gt)

    if args.data == 1:
        gt2file = glob.glob('data/Trento/' + key + '*gt*[0-9].npy')[0]
        gt2 = np.load(gt2file)
    elif args.data == 2:
        gt2file = glob.glob('data/Houston/' + key + '*gt*[0-9].npy')[0]
        gt2 = np.load(gt2file)
    else:
        pathgt2 = 'data/MUUFL7891011/muufl_gt12.mat'
        gt2 = loadmat(pathgt2)['label']


    gt_new[np.logical_and(gt_new == 0, gt2 != 0)] = unknown
    cfm,oa,aa,kappa,osr = rscls.gtcfm(pre_gsrl, gt_new, unknown)

    OSR[Running] = osr
    acc[Running] = oa
    A[Running] = aa
    k[Running] = kappa

    pre_to_draw = deepcopy(pre_gsrl)
    pre_to_draw[pre_to_draw == unknown] = 0
    imgDraw(pre_to_draw, spath + key + '_gsrl', path='./', show=False)



AAMean = np.mean(A)
AAStd = np.std(A)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)
OSRMean = np.mean(OSR)
OSRStd = np.std(OSR)
# print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
# print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OSR: " + "{:.4f}".format(100 * OSRMean) + " +- " + "{:.4f}".format(100 * OSRStd))
print ("average OA: " + "{:.4f}".format(100 * OAMean) + " +- " + "{:.4f}".format(100 * OAStd))
print ("average AA: " + "{:.4f}".format(100 * AAMean) + " +- " + "{:.4f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
# print ("accuracy for each class: ")
# for i in range(num_classes):
#     print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

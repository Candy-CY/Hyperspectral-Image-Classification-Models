# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:08:19 2020

@author: HLB
"""

import torch
import numpy as np
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

ce_loss = nn.CrossEntropyLoss()


def tr_acc(model, image, label):
    train_dataset = TensorDataset(torch.tensor(image), torch.tensor(label))
    train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = False)
    train_loss = 0
    corr_num = 0
    for idx, (image_batch, label_batch) in enumerate(train_loader):        
        trans_image_batch = image_batch.cuda()
        label_batch = label_batch.cuda()
        logits = model(trans_image_batch)
        if isinstance(logits,tuple):
            logits = logits[-1]
        pred = torch.max(logits, dim=1)[1]
        loss = ce_loss(logits, label_batch)                
        train_loss = train_loss + loss.cpu().data.numpy()
        corr_num = torch.eq(pred, label_batch).float().sum().cpu().numpy() + corr_num   
    return corr_num/image.shape[0], train_loss/(idx+1)

def pre_train(model, train_image, train_label,validation_image, validation_label, epoch, optimizer, scheduler, bs, val = False):        
    train_dataset = TensorDataset(torch.tensor(train_image), torch.tensor(train_label))
    train_loader = DataLoader(dataset = train_dataset, batch_size = bs, shuffle = True)
    Train_loss = []
    Train_acc = []
    Val_loss = []
    Val_acc = []
    for i in range(epoch): 
        model.train()
        train_loss = 0
        for idx, (image_batch, label_batch) in enumerate(train_loader):        

            trans_image_batch = image_batch.cuda()
            label_batch = label_batch.cuda()    
            logits = model(trans_image_batch)
            loss = ce_loss(logits, label_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + loss
        scheduler.step()
        train_loss = train_loss / (idx + 1)
        train_acc, tr_loss = tr_acc(model.eval(), train_image, train_label)
        val_acc, val_loss= tr_acc(model.eval(), validation_image, validation_label)
        print("epoch {}, training loss: {:.4f}, train acc:{:.4f}, valid acc:{:.4f}".format(i, train_loss.item(), train_acc*100, val_acc*100))

        if val:
            Train_loss.append(tr_loss)
            Val_loss.append(val_loss) 
            Train_acc.append(train_acc)
            Val_acc.append(val_acc)
    if val:
        return model,  [Train_loss, Train_acc, Val_loss, Val_acc]
    else:                  
        return model

def test_batch(model, image, index, BATCH_SIZE,  nTrain_perClass, nvalid_perClass, halfsize):
    ind = index[0][nTrain_perClass[0]+ nvalid_perClass[0]:,:]
    nclass = len(index)
    true_label = np.zeros(ind.shape[0], dtype = np.int32)
    for i in range(1, nclass):
        ddd = index[i][nTrain_perClass[i] + nvalid_perClass[i]:,:]
        ind = np.concatenate((ind, ddd), axis = 0)
        tr_label = np.ones(ddd.shape[0], dtype = np.int32) * i
        true_label = np.concatenate((true_label, tr_label), axis = 0)
    test_index = np.copy(ind)
    length = ind.shape[0]
    if length % BATCH_SIZE != 0:
        add_num = BATCH_SIZE - length % BATCH_SIZE        
        ff = range(length)    
        add_ind = np.random.choice(ff, add_num, replace = False)
        add_ind = ind[add_ind]        
        ind = np.concatenate((ind,add_ind), axis =0)

    pred_array = np.zeros([ind.shape[0],nclass], dtype = np.float32)
    n = ind.shape[0] // BATCH_SIZE
    windowsize = 2 * halfsize + 1
    image_batch = np.zeros([BATCH_SIZE, windowsize, windowsize, image.shape[2]], dtype=np.float32)
    for i in range(n):
        for j in range(BATCH_SIZE):
            m = ind[BATCH_SIZE*i+j, :]
            image_batch[j,:,:,:] = image[(m[0] - halfsize):(m[0] + halfsize + 1),
                                                   (m[1] - halfsize):(m[1] + halfsize + 1),:]
            image_b = np.transpose(image_batch,(0,3,1,2))
        logits = model(torch.tensor(image_b).cuda()) 
        if isinstance(logits,tuple):
            logits = logits[-1]            
        pred_array[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = torch.softmax(logits, dim = 1).cpu().data.numpy()
    pred_array = pred_array[range(length)]
    predict_label  = np.argmax(pred_array, axis=1)
    
    
    confusion_matrix = metrics.confusion_matrix(true_label, predict_label)
    overall_accuracy = metrics.accuracy_score(true_label, predict_label)
    
    true_cla = np.zeros(nclass,  dtype=np.int64)
    for i in range(nclass):
        true_cla[i] = confusion_matrix[i,i]
    test_num_class = np.sum(confusion_matrix,1)
    test_num = np.sum(test_num_class)
    num1 = np.sum(confusion_matrix,0)
    po = overall_accuracy
    pe = np.sum(test_num_class*num1)/(test_num*test_num)
    kappa = (po-pe)/(1-pe)*100
    true_cla = np.true_divide(true_cla,test_num_class)*100 
    average_accuracy = np.average(true_cla)
    print('overall_accuracy: {0:f}'.format(overall_accuracy*100)) 
    print('average_accuracy: {0:f}'.format(average_accuracy))  
    print('kappa:{0:f}'.format(kappa))
    return true_cla, overall_accuracy*100, average_accuracy, kappa, true_label, predict_label, test_index, confusion_matrix, pred_array

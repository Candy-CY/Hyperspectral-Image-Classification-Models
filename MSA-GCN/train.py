# -*- coding:utf-8 -*-

import torch
import torch.optim as optim
from module import *
import time
import matplotlib.pyplot as plt
import numpy as np
from enhance import *
from test import test_demo



def train_best_model(dataset, train_iter, device, epoches, ITER, TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, test_iter, num_classes, in_channels_2, windowSize,applyPCA_Channel):
    """
    if(dataset=="MUUFL"):
        step_size=100
        gamma=0.8
    else:
        step_size = 300
        gamma = 0.5
    """
    for index_iter in range(ITER):
        train_loss_list = []
        train_acc_list = []
        oa_list = []
        oa_need_list = []
        epoch_need_list = []
        x_epoch = []
        weight=[]
        eye = torch.eye(int(max(train_iter.dataset.tensors[6]) + 1)).cuda()
        net1 = Feature_HSI_Lidar(applyPCA_Channel, in_channels_2, num_classes, windowSize).to(device)

        # net.apply(weight_init)  #网络权重初始化
        optimizer1 = optim.Adam(net1.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
        # lr_adjust = torch.optim.lr_scheduler.StepLR(optimizer=optimizer1, step_size=300, gamma=0.5, last_epoch=-1)
        net1.train()
        # loss2 = torch.nn.MSELoss(reduction='mean')
        #　loss1 = nn.MSELoss()
        loss1 = nn.CrossEntropyLoss()
        print('\niter:', index_iter + 1)
        print('TRAIN_SIZE: ', TRAIN_SIZE)
        print('TEST_SIZE: ', TEST_SIZE)
        print('TOTAL_SIZE: ', TOTAL_SIZE)
        print(
            '--------------------------------------------------Training on {}--------------------------------------------------\n'.format(
                device))
        start = time.time()


        for epoch in range(epoches):
            train_acc_sum, train_loss_sum = 0.0, 0.0

            time_epoch = time.time()
            # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0.0, last_epoch=-1)

            for step, (X_hsi, h1, h2, X_lidar, l1, l2, target, position) in enumerate(train_iter):
                X_hsi = X_hsi.to(device)
                h1 = h1.to(device)
                h2 = h2.to(device)
                X_lidar = X_lidar.to(device)
                l1 = l1.to(device)
                l2 = l2.to(device)
                target = target.to(device)
                target = target.to(torch.int64)
                target_hot = eye[target]
                out = net1(X_hsi, h1, h2, X_lidar, l1, l2)

                l1 = loss1(out, target)
                # l1 = loss2(weight[0]*out1,X_hsi)+loss2(weight[1]*out2,X_lidar)
                # print(beta)
                # 反向传播及优化
                optimizer1.zero_grad()  # 梯度清零
                l1.backward()
                optimizer1.step()
                train_loss_sum += l1.cpu().item()
                train_acc_sum += (out.argmax(dim=-1) == target_hot.argmax(dim=-1).to(device)).float().sum().cpu().item()
            # lr_adjust.step()
            print(optimizer1.state_dict()['param_groups'][0]['lr'])
            print('epoch %d, train loss %.6f, train acc %.4f, time %.2f sec' % (
                epoch + 1, train_loss_sum / len(train_iter.dataset), train_acc_sum / len(train_iter.dataset),
                time.time() - time_epoch))

            train_loss_list.append(train_loss_sum / len(train_iter.dataset))  # / batch_count)

            if train_loss_list[-1] <= min(train_loss_list):
                torch.save(net1.state_dict(), './models/' + dataset + '.pt')
                print('**Successfully Saved Best hsi model parametres!***\n')  # 保存在训练集上损失值最好的模型效果

            if epoch % 50 == 0 and epoch != 0:
                oa = test_demo(test_iter=test_iter,
                               device=device,
                               dataset=dataset,
                               net1=net1)
                print(oa)
                oa_list.append(oa)
                print(oa_list)

        End = time.time()
        print(max(oa_list))
        print('***Training End! Total Time %.1f sec***' % (End - start))
        print(oa_list)

    return net1



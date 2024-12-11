# -*- coding:utf-8 -*-

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import time
import numpy as np
from module import *
from einops import rearrange

def test(test_iter, dataset, device, net1):
    net1.load_state_dict(torch.load('./models/' + dataset + '.pt'))  # 加载保存好的模型

    print('\n***Start  Testing***\n')
    tick1 = time.time()
    y_test = []
    y_pred = []
    with torch.no_grad():


        for step, (X_hsi, h1, h2, X_lidar, l1, l2, target, position) in enumerate(test_iter):
            net1.eval()
            X_hsi = X_hsi.to(device)
            h1 = h1.to(device)
            h2 = h2.to(device)
            X_lidar = X_lidar.to(device)
            l1 = l1.to(device)
            l2 = l2.to(device)
            target = target.to(device)
            out = net1(X_hsi, h1, h2, X_lidar, l1, l2)
            y_pred.extend(out.cpu().argmax(dim=1))
            y_test.extend(target.cpu())
            net1.train()


    tick2 = time.time()
    Test_time = tick2 - tick1
    if dataset == 'Houston':
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Tree',
                    'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway',
                    'Railway', 'Parking lot 1', 'Parking lot 2', 'Tennis court', 'Running track']
    if dataset == 'Berlin':
        target_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil',
                        'Allotment', 'Commercial Area', 'Water']
    if dataset == 'Trento':
        target_names = ['Apple trees', 'Buildings', 'Ground', 'Wood', 'Vineyard', 'Roads']
    if dataset == 'MUUFL':
        target_names = ['Trees', 'Mostly grass', 'Mixed ground surface', 'Dirt and sand', 'Road', 'Water',
                        'Building shadow', 'Building', 'Sidewalk', 'Yellow curb', 'Cloth panels']
    classification = classification_report(np.array(y_test), np.array(y_pred), target_names=target_names, digits=4)
    oa = accuracy_score(np.array(y_test), np.array(y_pred))
    confusion = confusion_matrix(np.array(y_test), np.array(y_pred))
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.array(y_test), np.array(y_pred))
    return classification, oa, aa, kappa, each_acc


def test_demo(test_iter, dataset, device, net1):
    net1.load_state_dict(torch.load('./models/' + dataset + '.pt'))  # 加载保存好的模型
    # print('\n***Start  Testing***\n')
    test_acc_sum = 0.0
    y_test = []
    y_pred = []
    with torch.no_grad():

        for step, (X_hsi, h1, h2, X_lidar, l1, l2, target, position) in enumerate(test_iter):
            net1.eval()
            X_hsi = X_hsi.to(device)
            h1 = h1.to(device)
            h2 = h2.to(device)
            X_lidar = X_lidar.to(device)
            l1 = l1.to(device)
            l2 = l2.to(device)
            target = target.to(device)
            out = net1(X_hsi, h1, h2, X_lidar, l1, l2)
            y_pred.extend(out.cpu().argmax(dim=1))
            y_test.extend(target.cpu())
            net1.train()
            test_acc_sum += (out.argmax(dim=-1) == target.to(device)).float().sum().cpu().item()


    oa = test_acc_sum / len(test_iter.dataset)
    return oa

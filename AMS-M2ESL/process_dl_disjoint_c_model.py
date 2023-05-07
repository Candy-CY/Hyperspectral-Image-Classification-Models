# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : process_dl_disjoint_c_model.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

import os
import time
import torch
import random
import pandas as pd
import numpy as np
from sklearn import metrics

from ptflops import get_model_complexity_info

import utils.evaluation as evaluation
import utils.data_load_operate_c_model as data_load_operate
import visual.cls_visual as cls_visual

import c_model.CNN1D as CNN1D
import c_model.MCM_CNN as MCM_CNN
import c_model.SSTN as SSTN
import c_model.SSSAN as SSSAN
import c_model.SSAtt as SSAtt
import c_model.A2S2KResNet as A2S2KResNet
import c_model.SSFTT as SSFTT
import c_model.ASPN as ASPN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eachClassLabelNum(n):
    # x=pd.DataFrame(n).value_counts(ascending=True)
    x = pd.DataFrame(n).value_counts()
    print(x)


time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())

# random seed setting

seed = 20

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#                  0         1         2       3        4           5           6        7
model_list = ['1DCNN', 'MCM-CNN', 'SSTN', 'SSSAN', 'SSAtt', 'A2S2KResNet', 'SSFTT', 'ASPN']

model_flag = 4
model_spa_set = {1, 2, 4, 5, 6, 7}
model_spe_set = {0}
model_spa_spe_set = {3}
model_3D_spa_set = {5, 6}
model_3D_spa_flag = 0

last_batch_flag = 0

if model_flag in model_spa_set:
    model_type_flag = 1
    if model_flag in model_3D_spa_set:
        model_3D_spa_flag = 1
elif model_flag in model_spe_set:
    model_type_flag = 2
elif model_flag in model_spa_spe_set:
    model_type_flag = 3

data_set_name_list = ['UH_tif']
data_set_name = data_set_name_list[0]

data_set_path = os.path.join(os.getcwd(), 'data')

# control running times
# seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# seed_list=[0,1,2,3,4]
# seed_list=[0,1,2]
# seed_list=[0,1]
seed_list = [0]

# data set split
ratio = "hu13"

patch_size = 9
patch_length = 4

results_save_path = \
    os.path.join(os.path.join(os.getcwd(), 'output/results'), model_list[model_flag] + str("_") +
                 data_set_name + str("_") + str(time_current) + str("_seed_") + str(seed) + str("_ratio_") + str(
        ratio) + str("_patch_size_") + str(patch_size))
cls_map_save_path = \
    os.path.join(os.path.join(os.getcwd(), 'output/cls_maps'), model_list[model_flag] + str("_") +
                 data_set_name + str("_") + str(time_current) + str("_seed_") + str(seed) + str("_ratio_") + str(ratio))

if __name__ == '__main__':

    data, gt_train, gt_test = data_load_operate.load_HU_data(data_set_path)
    data = data_load_operate.standardization(data)

    if model_flag == 1:
        ratio = round(20 / data.shape[-1], 2)
        data = data_load_operate.HSI_MNF(data, MNF_ratio=ratio)
    elif model_flag == 5:
        data = data_load_operate.applyPCA(data, data.shape[-1])
    elif model_flag == 6:
        data = data_load_operate.applyPCA(data, 30)
    elif model_flag == 7:
        data = data_load_operate.applyPCA(data, data.shape[-1])
        last_batch_flag = 1

    gt_train_re = gt_train.reshape(-1)
    gt_test_re = gt_test.reshape(-1)
    height, width, channels = data.shape
    class_count = max(np.unique(gt_train_re))

    dr_channels = 0
    if model_flag == 0:
        batch_size = 65
        learning_rate = 1e-3
    elif model_flag == 1:
        batch_size = 100
        learning_rate = 1e-3
    elif model_flag == 2:
        batch_size = 50
        learning_rate = 1e-3
    elif model_flag == 3:
        batch_size = 100
        learning_rate = 1e-3
        if data_set_name == 'IP':
            dr_channels = 60
        elif data_set_name == 'UP':
            dr_channels = 4
    elif model_flag == 4:
        batch_size = 128
        learning_rate = 1e-3
    elif model_flag == 5:
        batch_size = 32
        learning_rate = 1e-3
    elif model_flag == 6:
        batch_size = 64
        learning_rate = 1e-3
    elif model_flag == 7:
        batch_size = 64
        learning_rate = 1e-3

    scales = 15
    max_epoch = 40
    loss = torch.nn.CrossEntropyLoss()

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])

    # data pad zero
    # data:[h,w,c]->data_padded:[h+2l,w+2l,c]
    data_padded = data_load_operate.data_pad_zero(data, patch_length)
    height_patched, width_patched, channels = data_padded.shape

    data_total_index = np.arange(data.shape[0] * data.shape[1])  # For total sample cls_map.

    for curr_seed in seed_list:
        tic1 = time.perf_counter()

        # wo val
        train_data_index, test_data_index, all_data_index = data_load_operate.sampling_disjoint(gt_train_re,
                                                                                                gt_test_re,
                                                                                                class_count)
        index = (train_data_index, test_data_index)
        train_iter, test_iter = data_load_operate.generate_iter_disjoint(data_padded, data, height, width,
                                                                         gt_train_re,
                                                                         gt_test_re, index, patch_length,
                                                                         batch_size, model_type_flag,
                                                                         model_3D_spa_flag, last_batch_flag,
                                                                         model_flag, dr_channels, scales)

        # total_iter = data_load_operate.generate_iter_total(data_padded, data, height, width, gt_train_re, data_total_index,
        #                                                patch_length,
        #                                                batch_size, model_type_flag, model_3D_spa_flag, last_batch_flag,
        #                                                model_flag,
        #                                                dr_channels, scales)

        if model_flag == 0:
            net = CNN1D.CNN1D(channels, class_count)
        elif model_flag == 1:
            net = MCM_CNN.MCM_CNN_(scales, class_count, data_set_name)
        elif model_flag == 2:
            net = SSTN.SSTN_AEAE(in_shape=(channels, height_patched, width_patched), num_classes=class_count)
        elif model_flag == 3:
            net = SSSAN.SSSAN(channels, dr_channels, class_count)
        elif model_flag == 4:
            net = SSAtt.Hang2020(channels, class_count)
        elif model_flag == 5:
            net = A2S2KResNet.S3KAIResNet(channels, class_count, 2)
        elif model_flag == 6:
            net = SSFTT.SSFTTnet(num_classes=class_count)
        elif model_flag == 7:
            net = ASPN.ASPN_(batch_size, patch_size, patch_size, channels, class_count)

        net.to(device)

        # efficiency test, model complexity and computational cost
        # flops,para=get_model_complexity_info(net,(channels,1,1),as_strings=False,print_per_layer_stat=True, verbose=True)
        # flops,para=get_model_complexity_info(net,(patch_size,patch_size,channels),as_strings=False,print_per_layer_stat=True, verbose=True)
        # flops,para=get_model_complexity_info(net,(scales,channels,channels),as_strings=False,print_per_layer_stat=True, verbose=True)
        # flops,para=get_model_complexity_info(net,(1,patch_size,patch_size,channels),as_strings=False,print_per_layer_stat=True, verbose=True)
        # print("para(M):{:.3f},\n flops(M):{:.3f}".format(para/(1000**2),flops/(1000**2)))

        train_loss_list = [100]
        train_acc_list = [0]

        if model_flag == 0 or model_flag == 2 or model_flag == 4 or model_flag == 6:
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        elif model_flag == 1 or model_flag == 3:
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)
        elif model_flag == 5:
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                         weight_decay=0)
            lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 15, eta_min=0.0, last_epoch=-1)
        elif model_flag == 7:
            optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=learning_rate, eps=1e-5)

        for epoch in range(max_epoch):
            train_acc_sum, trained_samples_counter = 0.0, 0
            batch_counter, train_loss_sum = 0, 0
            time_epoch = time.time()

            if model_type_flag == 1:  # data for single spatial net
                for X_spa, y in train_iter:
                    X_spa, y = X_spa.to(device), y.to(device)
                    y_pred = net(X_spa)

                    ls = loss(y_pred, y.long())

                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
                    epoch_first_iter = 0
                if model_flag == 5:
                    lr_adjust.step()
            elif model_type_flag == 2:  # data for single spectral net
                for X_spe, y in train_iter:
                    X_spe, y = X_spe.to(device), y.to(device)
                    y_pred = net(X_spe)

                    ls = loss(y_pred, y.long())

                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
                    epoch_first_iter = 0
            elif model_type_flag == 3:  # data for spectral-spatial net
                for X_spa, X_spe, y in train_iter:
                    X_spa, X_spe, y = X_spa.to(device), X_spe.to(device), y.to(device)
                    y_pred = net(X_spa, X_spe)

                    ls = loss(y_pred, y.long())

                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
                    epoch_first_iter = 0

            torch.cuda.empty_cache()

            train_loss_list.append(train_loss_sum)
            train_acc_list.append(train_acc_sum / trained_samples_counter)

            print('epoch: %d, training_sampler_num: %d, batch_count: %.2f, train loss: %.6f, tarin loss sum: %.6f, '
                  'train acc: %.3f, train_acc_sum: %.1f, time: %.1f sec' %
                  (epoch + 1, trained_samples_counter, batch_counter, train_loss_sum / batch_counter, train_loss_sum,
                   train_acc_sum / trained_samples_counter, train_acc_sum, time.time() - time_epoch))

        toc1 = time.perf_counter()
        print('Training stage finished:\n epoch %d, loss %.4f, train acc %.3f, training time %.2f s'
              % (epoch + 1, train_loss_sum / batch_counter, train_acc_sum / trained_samples_counter, toc1 - tic1))
        training_time = toc1 - tic1
        Train_Time_ALL.append(training_time)

        print("\n\n====================Starting evaluation for testing set.========================\n")

        pred_test = []
        y_gt = []
        # torch.cuda.empty_cache()
        with torch.no_grad():
            # net.load_state_dict(torch.load(model_save_path+"_best_model.pt"))
            net.eval()
            train_acc_sum, samples_num_counter = 0.0, 0
            if model_type_flag == 1:  # data for single spatial net
                for X_spa, y in test_iter:
                    X_spa = X_spa.to(device)

                    tic2 = time.perf_counter()
                    y_pred = net(X_spa)
                    toc2 = time.perf_counter()

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))
                    y_gt.extend(y)
            elif model_type_flag == 2:  # data for single spectral net
                for X_spe, y in test_iter:
                    X_spe = X_spe.to(device)

                    tic2 = time.perf_counter()
                    y_pred = net(X_spe)
                    toc2 = time.perf_counter()

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))
                    y_gt.extend(y)
            elif model_type_flag == 3:  # data for spectral-spatial net
                for X_spa, X_spe, y in test_iter:
                    X_spa = X_spa.to(device)
                    X_spe = X_spe.to(device)

                    tic2 = time.perf_counter()
                    y_pred = net(X_spa, X_spe)
                    toc2 = time.perf_counter()

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))
                    y_gt.extend(y)
            if last_batch_flag == 0:
                y_gt = gt_test_re[test_data_index] - 1
            OA = metrics.accuracy_score(y_gt, pred_test)
            confusion_matrix = metrics.confusion_matrix(pred_test, y_gt)
            print("confusion_matrix\n{}".format(confusion_matrix))
            ECA, AA = evaluation.AA_ECA(confusion_matrix)
            kappa = metrics.cohen_kappa_score(pred_test, y_gt)
            cls_report = evaluation.claification_report(y_gt, pred_test, data_set_name)
            print("classification_report\n{}".format(cls_report))

            # Visualization for all the labeled samples and total the samples
            # sample_list1 = [total_iter]
            # sample_list2 = [all_iter, all_data_index]

            # Visualization.gt_cls_map(gt,cls_map_save_path)
            # cls_visual.pred_cls_map_dl(sample_list1, net, gt_train, cls_map_save_path, model_type_flag)
            # cls_visual.pred_cls_map_dl(sample_list2,net,gt,cls_map_save_path)

            testing_time = toc2 - tic2
            Test_Time_ALL.append(testing_time)

            # Output infors
            f = open(results_save_path + '_results.txt', 'a+')
            str_results = '\n======================' \
                          + " learning rate=" + str(learning_rate) \
                          + " epochs=" + str(max_epoch) \
                          + " ======================" \
                          + "\nOA=" + str(OA) \
                          + "\nAA=" + str(AA) \
                          + '\nkpp=' + str(kappa) \
                          + '\nacc per class:' + str(ECA) \
                          + "\ntrain time:" + str(training_time) \
                          + "\ntest time:" + str(testing_time) + "\n"

            f.write(str_results)
            f.write('{}'.format(confusion_matrix))
            f.write('\n\n')
            f.write('{}'.format(cls_report))
            f.close()

            OA_ALL.append(OA)
            AA_ALL.append(AA)
            KPP_ALL.append(kappa)
            EACH_ACC_ALL.append(ECA)

        torch.cuda.empty_cache()
        del net, train_iter, test_iter

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    print("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    print('List of OA:', list(OA_ALL))
    print('List of AA:', list(AA_ALL))
    print('List of KPP:', list(KPP_ALL))
    print('OA=', round(np.mean(OA_ALL) * 100, 2), '+-', round(np.std(OA_ALL) * 100, 2))
    print('AA=', round(np.mean(AA_ALL) * 100, 2), '+-', round(np.std(AA_ALL) * 100, 2))
    print('Kpp=', round(np.mean(KPP_ALL) * 100, 2), '+-', round(np.std(KPP_ALL) * 100, 2))
    print('Acc per class=', np.mean(EACH_ACC_ALL, 0), '+-', np.std(EACH_ACC_ALL, 0))

    print("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    print("Average testing time=", round(np.mean(Test_Time_ALL), 5), '+-', round(np.std(Test_Time_ALL), 5))

    # Output infors
    f = open(results_save_path + '_results.txt', 'a+')
    str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + 'times runs ********************' \
                  + '\nList of OA:' + str(list(OA_ALL)) \
                  + '\nList of AA:' + str(list(AA_ALL)) \
                  + '\nList of KPP:' + str(list(KPP_ALL)) \
                  + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                  + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                  + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(round(np.std(KPP_ALL) * 100, 2)) \
                  + '\nAcc per class=\n' + str(np.mean(EACH_ACC_ALL, 0)) + '+-' + str(np.std(EACH_ACC_ALL, 0)) \
                  + "\nAverage training time=" + str(round(np.mean(Train_Time_ALL), 2)) + '+-' + str(
        round(np.std(Train_Time_ALL), 3)) \
                  + "\nAverage testing time=" + str(round(np.mean(Test_Time_ALL), 5)) + '+-' + str(
        round(np.std(Test_Time_ALL), 5))
    f.write(str_results)
    f.close()

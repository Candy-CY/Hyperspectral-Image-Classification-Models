# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2022-Nov
# @Address  : Time Lab @ SDU
# @FileName : process_dl_disjoint.py
# @Project  : CVSSN (HSIC), IEEE TCSVT


# for spatially disjoint data set, e.g., the UH data set,
# main processing file for the involved deep learning models,
# i.e., ,
# ContextualNet, RSSAN, SSTN, SSAtt, SSAN, SSSAN, A2S2KResNet, and the proposed CVSSN

import os
import time
import torch
import random
import numpy as np
import pandas as pd
from sklearn import metrics

import utils.evaluation as evaluation
import utils.data_load_operate as data_load_operate
import utils.data_load_operate_disjoint as data_load_operate_disjoint

import model.ContextualNet as ContextualNet
import model.RSSAN as RSSAN
import model.SSTN as SSTN
import model.SSAtt as SSAtt
import model.A2S2KResNet as A2S2KResNet
import model.CVSSN as CVSSN
import model.SSAN as SSAN
import model.SSSAN as SSSAN

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###                 0             1       2       3        4        5         6             7
model_list = ['ContextualNet', 'RSSAN', 'SSTN', 'SSAN', 'SSSAN', 'SSAtt', 'A2S2KResNet', 'CVSSN']

model_flag = 7
model_spa_set = {1, 2, 3, 5}
model_spe_set = {}
model_spa_spe_set = {4, 7}
model_3D_spa_set = {0, 6}

model_3D_spa_flag = 0

if model_flag in model_spa_set:
    model_type_flag = 1
    if model_flag in model_3D_spa_set:
        model_3D_spa_flag = 1
elif model_flag in model_spe_set:
    model_type_flag = 2
elif model_flag in model_spa_spe_set:
    model_type_flag = 3

# 3
data_set_name_list = ['IP', 'KSC', 'UP', 'HU_tif']
data_set_name = data_set_name_list[3]

# seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  #
# seed_list=[0,1,2,3,4]
# seed_list=[0,1,2] #
# seed_list=[0,1]
seed_list = [0]  #

# ratio=1.0
# ratio=5.0
# ratio=7.5
# ratio = 10.0
ratio = "hu13"

data_set_path = os.path.join(os.getcwd(), 'data')

results_save_path = \
    os.path.join(os.path.join(os.getcwd(), 'output/results'), model_list[model_flag] + str("_") +
                 data_set_name + str("_") + str(time_current) + str("_seed") + str(seed)) + str("_ratio") + str(ratio)
cls_map_save_path = \
    os.path.join(os.path.join(os.getcwd(), 'output/cls_maps'), model_list[model_flag] + str("_") +
                 data_set_name + str("_") + str(time_current) + str("_seed") + str(seed)) + str("_ratio") + str(ratio)

if __name__ == '__main__':

    torch.cuda.empty_cache()

    data, gt_train, gt_test = data_load_operate.load_HU_data(data_set_name, data_set_path)

    data = data_load_operate.standardization(data)

    gt_train_re = gt_train.reshape(-1)
    gt_test_re = gt_test.reshape(-1)
    height, width, channels = data.shape
    class_count = max(np.unique(gt_train_re))

    patch_size = 9
    patch_length = 4
    batch_size = 32
    max_epoch = 10
    learning_rate = 0.001
    loss = torch.nn.CrossEntropyLoss()

    # data pad zero
    # data:[h,w,c]->data_padded:[h+2l,w+2l,c]
    data_padded = data_load_operate.data_pad_zero(data, patch_length)
    height_patched, width_patched, channels = data_padded.shape

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])

    # data_total_index = np.arange(data.shape[0] * data.shape[1])  # For total sample cls_map.

    for curr_seed in seed_list:

        # # w val
        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate_disjoint.sampling_UH_w_val(
            gt_train_re,
            gt_test_re,
            class_count)

        index = (train_data_index, val_data_index, test_data_index)
        train_iter, val_iter, test_iter = data_load_operate_disjoint.generate_iter_hu_w_val(data_padded, data, gt_train_re,
                                                                                         gt_test_re, index,
                                                                                         patch_length, batch_size,
                                                                                         model_type_flag,
                                                                                         model_3D_spa_flag)

        # all_iter = data_load_operate.generate_iter_2(data_padded, data, gt_reshape, all_data_index, patch_length,
        #                                              batch_size, model_type_flag, model_3D_spa_flag)
        # total_iter = data_load_operate.generate_iter_2(data_padded, data, gt_reshape, data_total_index, patch_length,
        #                                                25, model_type_flag, model_3D_spa_flag)

        if model_flag == 0:
            net = ContextualNet.LeeEtAl(channels, class_count)
        elif model_flag == 1:
            net = RSSAN.RSSAN_net(in_shape=(channels, height_patched, width_patched), num_classes=class_count)
        elif model_flag == 2:
            net = SSTN.SSTN_AEAE(in_shape=(channels, height_patched, width_patched), num_classes=class_count)
        elif model_flag == 3:
            net = SSAN.SSAN(channels, patch_size, class_count)
        elif model_flag == 4:
            net = SSSAN.SSSAN(channels, class_count)
        elif model_flag == 5:
            net = SSAtt.Hang2020(channels, class_count)
        elif model_flag == 6:
            net = A2S2KResNet.S3KAIResNet(channels, class_count, 2)
        elif model_flag == 7:
            net = CVSSN.CVSSN_(channels, patch_size, patch_size, class_count)

        # efficiency test, model complexity and computational cost
        # test_spe_input=torch.randn(1,channels) # for 1D model
        # test_input=torch.randn(1,patch_size,patch_size,channels) # for 2D model
        # test_input=torch.randn(1,1,patch_size,patch_size,channels) # for 3D model
        #
        # flops,para=profile(net,(test_input,test_spe_input))
        # flops,para=profile(net,(test_spe_input))
        # flops,para=profile(net,(test_input))
        #
        # print("para:{}\n,flops:{}".format(para,flops))
        # print("para(M):{:.3f},\n flops(M):{:.2f}".format(para/(1000**2),flops/(1000**2),))

        net.to(device)

        train_loss_list = [100]
        train_acc_list = [0]
        val_loss_list = [100]
        val_acc_list = [0]
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        best_loss = 99999

        tic1 = time.perf_counter()
        for epoch in range(max_epoch):
            train_acc_sum, trained_samples_counter = 0.0, 0
            batch_counter, train_loss_sum = 0, 0
            net.train()
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
                    if model_flag == 10:
                        for i in range(len(y_pred)):
                            if i == 0:
                                ls = loss(y_pred[i], y.long())
                            if i > 0:
                                ls += loss(y_pred[i], y.long())
                    else:

                        ls = loss(y_pred, y.long())

                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
                    epoch_first_iter = 0

            val_acc, val_loss = evaluation.evaluate_OA(val_iter, net, loss, device, model_type_flag)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(net.state_dict(), results_save_path + "_best_model.pt")
                print('save model...')

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
        # torch.cuda.empty_cache()
        with torch.no_grad():
            # net.load_state_dict(torch.load(model_save_path+"_best_model.pt"))
            net.eval()
            train_acc_sum, samples_num_counter = 0.0, 0

            if model_type_flag == 1:  # data for single spatial net
                for X_spa, y in test_iter:
                    X_spa = X_spa.to(device)
                    y = y.to(device)

                    tic2 = time.perf_counter()
                    y_pred = net(X_spa)
                    toc2 = time.perf_counter()

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))
            elif model_type_flag == 2:  # data for single spectral net
                for X_spe, y in test_iter:
                    X_spe = X_spe.to(device)
                    y = y.to(device)

                    tic2 = time.perf_counter()
                    y_pred = net(X_spe)
                    toc2 = time.perf_counter()

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))
            elif model_type_flag == 3:  # data for spectral-spatial net
                for X_spa, X_spe, y in test_iter:
                    X_spa = X_spa.to(device)
                    X_spe = X_spe.to(device)
                    y = y.to(device)

                    tic2 = time.perf_counter()
                    y_pred = net(X_spa, X_spe)
                    toc2 = time.perf_counter()

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))

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

            # cls_visual.pred_cls_map_dl(sample_list1,net,gt,cls_map_save_path)
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
        # del net, train_iter, test_iter, val_iter, all_iter
        # del net

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
    print('Acc per class=', np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2), '+-',
          np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2))

    print("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    print("Average testing time=", round(np.mean(Test_Time_ALL) * 1000, 2), '+-',
          round(np.std(Test_Time_ALL) * 1000, 3))

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
                  + "\nAverage training time=" + str(np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(
        np.round(np.std(Train_Time_ALL), decimals=3)) \
                  + "\nAverage testing time=" + str(np.round(np.mean(Test_Time_ALL) * 1000, decimals=2)) + '+-' + str(
        np.round(np.std(Test_Time_ALL) * 100, decimals=3))
    f.write(str_results)
    f.close()

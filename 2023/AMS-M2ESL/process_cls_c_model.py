# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Apr
# @Address  : Time Lab @ SDU
# @FileName : process_cls_c_model.py
# @Project  : AMS-M2ESL (HSIC), IEEE TGRS

# # for IP and UP data sets, main processing file for SVM-RBF

import os
import time
import torch
import random
import numpy as np

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import utils.evaluation as evaluation
import utils.data_load_operate as data_load_operate
import visual.cls_visual as cls_visual

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

########      0
model_list = ['SVM']
model_flag = 0

data_set_name_list = ['IP', 'UP', 'KSC', 'HU_tif']
data_set_name = data_set_name_list[1]

data_set_path = os.path.join(os.getcwd(), 'data')

# seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# seed_list=[0,1,2,3,4]
# seed_list=[0,1,2]
# seed_list=[0,1]
seed_list = [0]

# data set split
flag_list = [0, 1]  # ratio or num

if data_set_name == 'IP':
    ratio_list = [0.05, 0.005]
    ratio = 5.0
elif data_set_name == 'UP':
    ratio_list = [0.01, 0.001]
    ratio = 1.0

num_list = [50, 0]  # [train_num,val_num]

results_save_path = \
    os.path.join(os.path.join(os.getcwd(), 'output/results'), model_list[model_flag] + str("_") +
                 data_set_name + str("_") + str(time_current) + str("_seed_") + str(seed) + str("_ratio_") + str(ratio))
cls_map_save_path = \
    os.path.join(os.path.join(os.getcwd(), 'output/cls_maps'), model_list[model_flag] + str("_") +
                 data_set_name + str("_") + str(time_current) + str("_seed_") + str(seed) + str("_ratio_") + str(ratio))

if __name__ == '__main__':

    data, gt = data_load_operate.load_data(data_set_name, data_set_path)
    data = data_load_operate.standardization(data)

    gt_reshape = gt.reshape(-1)
    height, width, channels = data.shape
    class_count = max(np.unique(gt))

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])

    data_reshape = data.reshape(data.shape[0] * data.shape[1], -1)
    for curr_seed in seed_list:
        tic1 = time.perf_counter()

        train_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list, num_list,
                                                                                              gt_reshape,
                                                                                              class_count, flag_list[0])
        index = (train_data_index, test_data_index, all_data_index)
        x_train, y_train, x_test, y_gt, x_all, y_all = data_load_operate.generate_data_set(data_reshape, gt_reshape,
                                                                                           index)

        if model_flag == 0:
            clf = SVC(kernel='rbf', gamma='scale', C=20, tol=1e-5, random_state=10).fit(x_train, y_train)

        toc1 = time.perf_counter()
        training_time = toc1 - tic1
        Train_Time_ALL.append(training_time)

        tic2 = time.perf_counter()
        pred_test = clf.predict(x_test)
        toc2 = time.perf_counter()

        testing_time = toc2 - tic2
        Test_Time_ALL.append(testing_time)

        y_gt = gt_reshape[test_data_index] - 1
        OA = metrics.accuracy_score(y_gt, pred_test)
        confusion_matrix = metrics.confusion_matrix(pred_test, y_gt)
        print("confusion_matrix\n{}".format(confusion_matrix))
        ECA, AA = evaluation.AA_ECA(confusion_matrix)
        kappa = metrics.cohen_kappa_score(pred_test, y_gt)
        cls_report = evaluation.claification_report(y_gt, pred_test, data_set_name)
        print("classification_report\n{}".format(cls_report))

        # Visualization for all the labeled samples and total the samples
        # total_pred = clf.predict(data_reshape)
        # sample_list1 = [total_pred + 1]

        # all_pred=clf.predict(x_all)
        # sample_list2=[all_pred+1,all_data_index]

        # cls_visual.gt_cls_map(gt,cls_map_save_path)
        # cls_visual.pred_cls_map_cls(sample_list1,gt,cls_map_save_path)
        # cls_visual.pred_cls_map_cls(sample_list2,gt,cls_map_save_path)

        # Output infors
        f = open(results_save_path + '_results.txt', 'a+')
        str_results = '\n======================' \
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

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    print("\n====================Mean result of {} times runs ==========================".format(len(seed_list)))
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
                  + '\nAcc per class=' + str(np.mean(EACH_ACC_ALL, 0)) + '+-' + str(np.std(EACH_ACC_ALL, 0)) \
                  + "\nAverage training time=" + str(round(np.mean(Train_Time_ALL), 2)) + '+-' + str(
        round(np.std(Train_Time_ALL), 3)) \
                  + "\nAverage testing time=" + str(round(np.mean(Test_Time_ALL), 5)) + '+-' + str(
        round(np.std(Test_Time_ALL), 5))
    f.write(str_results)
    f.close()

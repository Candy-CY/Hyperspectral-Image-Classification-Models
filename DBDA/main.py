import numpy as np
import time
import collections
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime

import sys
sys.path.append('../global_module/')
import network
import train
from generate_pic import aa_and_each_accuracy, sampling,load_dataset, generate_png, generate_iter
from Utils import fdssc_model, record, extract_samll_cubic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for Monte Carlo runs
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')



global Dataset  # UP,IN,KSC
dataset = input('Please input the name of Dataset(IN, UP, BS, SV, PC or KSC):')
Dataset = dataset.upper()
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(Dataset)

print(data_hsi.shape)
image_x, image_y, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
CLASSES_NUM = max(gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = 10
PATCH_LENGTH = 4
# number of training samples per class
#lr, num_epochs, batch_size = 0.001, 200, 32
lr, num_epochs, batch_size = 0.00050, 200, 16
#lr, num_epochs, batch_size = 0.00050, 200, 16
#lr, num_epochs, batch_size = 0.0005, 200, 12
#net = network.DBDA_network_drop(BAND, CLASSES_NUM)
#net = network.DBDA_network_PReLU(BAND, CLASSES_NUM)
# net = network.DBMA_network(BAND, CLASSES_NUM)
# optimizer = optim.Adam(net.parameters(), lr=lr) #, weight_decay=0.0001)
loss = torch.nn.CrossEntropyLoss()

img_rows = 2*PATCH_LENGTH+1
img_cols = 2*PATCH_LENGTH+1
img_channels = data_hsi.shape[2]
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
VAL_SIZE = int(TRAIN_SIZE)
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE


KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

data = preprocessing.scale(data)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)

for index_iter in range(ITER):
    print('iter:', index_iter)
    net = network.DBDA_network_MISH(BAND, CLASSES_NUM)
    optimizer = optim.Adam(net.parameters(), lr=lr, amsgrad=False) #, weight_decay=0.0001)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    _, total_indices = sampling(1, gt)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----')

    train_iter, valida_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt)

    tic1 = time.clock()
    train.train(net, train_iter, valida_iter, loss, optimizer, device, epochs=num_epochs)
    toc1 = time.clock()

    pred_test_fdssc = []
    tic2 = time.clock()
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            net.eval()  # 评估模式, 这会关闭dropout
            y_hat = net(X)
            # print(net(X))
            pred_test_fdssc.extend(np.array(net(X).cpu().argmax(axis=1)))
    toc2 = time.clock()
    collections.Counter(pred_test_fdssc)
    gt_test = gt[test_indices] - 1


    overall_acc_fdssc = metrics.accuracy_score(pred_test_fdssc, gt_test[:-VAL_SIZE])
    confusion_matrix_fdssc = metrics.confusion_matrix(pred_test_fdssc, gt_test[:-VAL_SIZE])
    each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
    kappa = metrics.cohen_kappa_score(pred_test_fdssc, gt_test[:-VAL_SIZE])

    torch.save(net.state_dict(), "./net/" + str(round(overall_acc_fdssc, 3)) + '.pt')
    KAPPA.append(kappa)
    OA.append(overall_acc_fdssc)
    AA.append(average_acc_fdssc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc_fdssc

print("--------" + net.name + " Training Finished-----------")
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                     'records/' + net.name + day_str + '_' + Dataset + 'split：' + str(VALIDATION_SPLIT) + 'lr：' + str(lr) + '.txt')


generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices)

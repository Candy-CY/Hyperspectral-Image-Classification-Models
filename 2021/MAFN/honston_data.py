# -*- coding: utf-8 -*-
import collections
import time
import honston_net
import keras.callbacks as kcallbacks
import numpy as np
import scipy.io as sio
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from sklearn import metrics, preprocessing
from Utils import zeroPadding, averageAccuracy, Kappa
from matplotlib.colors import ListedColormap
from matplotlib import pyplot
from spectral import spy_colors, save_rgb

# sess=tf.Session()

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
#     m = max(groundTruth)
#     for i in range(m):
#         indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
#         np.random.shuffle(indices)
#
#         labels_loc[i] = indices
#         nb_val = int(proptionVal * len(indices))
#         #print(nb_val)
#         train[i] = indices[:-nb_val]
#         test[i] = indices[-nb_val:]
# #    whole_indices = []
#     train_indices = []
#     test_indices = []
#     for i in range(m):
# #        whole_indices += labels_lmodel_feature2.preoc[i]
#         train_indices += train[i]
#         test_indices += test[i]
#     np.random.shuffle(train_indices)
#     np.random.shuffle(test_indices)
#     return train_indices, test_indmodel_feature2.preices
#     print(len(test_indices))
    m=max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)

        labels_loc[i] = indices
        # nb_val = int(proptionVal * len(indices))
        # nb_val = proptionVal

        # train[i] = indices[:nb_val]
        # test[i] = indices[nb_val:]
        train[i]=indices
    train_indices = []
    # test_indices = []

    for i in range(m):
        train_indices += train[i]
        # test_indices += test[i]
    np.random.shuffle(train_indices)
    # np.random.shuffle(test_indices)
    # return train_indices, test_indices
    return train_indices
def sampling1(proptionVal, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}

    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)

        labels_loc[i] = indices

        nb_val = proptionVal

        # train[i] = indices[:nb_val]
        train[i]=indices

    train_indices = []


    for i in range(m):
        train_indices += train[i]

    np.random.shuffle(train_indices)

    return train_indices


mat_data = sio.loadmat('dataset/houston/houston.mat')
data_IN = mat_data['data']

mat_data = sio.loadmat('dataset/houston/houston_30.mat')
data_IN1 = mat_data['data']

mat_gt = sio.loadmat('dataset/houston/mask_train.mat')
gt_IN = mat_gt['mask_train']

mat_gt1 = sio.loadmat('dataset/houston/mask_test.mat')
gt_IN1 = mat_gt1['mask_test']

new_gt_IN = gt_IN

new_gt_IN1 = gt_IN1

batch_size = 16
nb_classes = 15
nb_epoch = 50

patience = 200
INPUT_DIMENSION_CONV = 144
INPUT_DIMENSION_CONV1 = 30

PATCH_LENGTH = 1
PATCH_LENGTH1 = 12


TOTAL_SIZE = 15029
VAL_SIZE = 520
# TRAIN_SIZE = 1715
TRAIN_SIZE= 2832
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.95
VALIDATION_SPLIT1 = 100

data = data_IN.reshape(np.prod(data_IN.shape[:2]),np.prod(data_IN.shape[2:]))
data1 = data_IN1.reshape(np.prod(data_IN1.shape[:2]), np.prod(data_IN1.shape[2:]))

gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)
gt1 = new_gt_IN1.reshape(np.prod(new_gt_IN1.shape[:2]),)

print(gt.shape)
data = preprocessing.scale(data)
data_ = data.reshape(data_IN.shape[0], data_IN.shape[1],data_IN.shape[2])

data1 = preprocessing.scale(data1)
data1_ = data1.reshape(data_IN1.shape[0], data_IN1.shape[1],data_IN1.shape[2])

whole_data = data_
whole_data1 = data1_

padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)
padded_data1 = zeroPadding.zeroPadding_3D(whole_data1, PATCH_LENGTH1)

ITER = 1
CATEGORY = 15

train_data = np.zeros((TRAIN_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
train_data1 = np.zeros((TRAIN_SIZE, 2*PATCH_LENGTH1 + 1, 2*PATCH_LENGTH1 + 1, INPUT_DIMENSION_CONV1))



test_data = np.zeros((TEST_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
test_data1 = np.zeros((TEST_SIZE, 2*PATCH_LENGTH1 + 1, 2*PATCH_LENGTH1 + 1, INPUT_DIMENSION_CONV1))

KAPPA_RES_SS4 = []
OA_RES_SS4 = []
AA_RES_SS4 = []
TRAINING_TIME_RES_SS4 = []
TESTING_TIME_RES_SS4 = []
ELEMENT_ACC_RES_SS4 = np.zeros((ITER, CATEGORY))
NUM=1
oa_all=np.zeros([1,NUM])
aa_all=np.zeros([1,NUM])
kappa_all=np.zeros([1,NUM])
seeds = [1221]

for num in range(NUM):
    for index_iter in range(ITER):
        print("# %d Iteration" % (index_iter + 1))

        # best_weights_RES_path_ss49 = 'models/UP_best_RES_3D_SS4_19_' + str(
        #     index_iter + 1) + '.hdf5'

        np.random.seed(seeds[index_iter])

        train_indices= sampling(VALIDATION_SPLIT, gt)
        test_indices = sampling1(VALIDATION_SPLIT1, gt1)


        print('train_indices', len(train_indices))
        print('test_indices',len(test_indices))

        y_train = gt[train_indices] - 1
        y_train = to_categorical(np.asarray(y_train))

        # y_test = gt[test_indices] - 1
        # y_test = to_categorical(np.asarray(y_test))

        y_test = gt1[test_indices] - 1
        y_test = to_categorical(np.asarray(y_test))


        train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
        train_assign1 = indexToAssignment(train_indices, whole_data1.shape[0], whole_data1.shape[1], PATCH_LENGTH1)


        for i in range(len(train_assign)):
            train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

            train_data1[i] = selectNeighboringPatch(padded_data1, train_assign1[i][0], train_assign1[i][1], PATCH_LENGTH1)


        test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
        test_assign1 = indexToAssignment(test_indices, whole_data1.shape[0], whole_data1.shape[1], PATCH_LENGTH1)
        # sess2=tf.Session()
        for i in range(len(test_assign)):
            test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)
            test_data1[i] = selectNeighboringPatch(padded_data1, test_assign1[i][0], test_assign1[i][1], PATCH_LENGTH1)


        x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
        x_train1= train_data1.reshape(train_data1.shape[0], train_data1.shape[1], train_data1.shape[2], INPUT_DIMENSION_CONV1)


        x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)
        x_test_all1 = test_data1.reshape(test_data1.shape[0], test_data1.shape[1], test_data1.shape[2], INPUT_DIMENSION_CONV1)


        x_val = x_test_all[-VAL_SIZE:]
        x_val1 = x_test_all1[-VAL_SIZE:]
        y_val = y_test[-VAL_SIZE:]

        x_test = x_test_all[:-VAL_SIZE]
        x_test1 = x_test_all1[:-VAL_SIZE]
        y_test = y_test[:-VAL_SIZE]
        test1_indices = test_indices[:-VAL_SIZE]
        print("x_train shape :", x_train.shape, x_train1.shape)
        print("y_train shape :", y_train.shape)
        print('x_val shape :', x_val.shape, x_val1.shape)
        print('y_val shape :', y_val.shape)
        print("x_test shape :", x_test.shape, x_test1.shape)
        print("y_test shape :", y_test.shape)


        model = honston_net.model()

        # earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
        # saveBestModel6 = kcallbacks.ModelCheckpoint(filepath='ckpt/houston.h5', monitor='val_loss', verbose=1,
        #                                             save_best_only=True,
        #                                             mode='auto')
        # print("----------------------------training------------------------------")
        # tic6 = time.clock()
        # history_mss_BN = model.fit(x=[x_train, x_train1], y=y_train,validation_data=([x_val, x_val1],y_val),
        #                          callbacks=[earlyStopping6, saveBestModel6],  batch_size=batch_size, epochs=nb_epoch, shuffle=True)
        # toc6 = time.clock()
        # print("----------------------------test now------------------------------")
        # tic7 = time.clock()
        # model.save("models/houston.h5")

        model.load_weights('ckpt/houston.h5')
        pred_test = model.predict([x_test, x_test1]).argmax(axis=1)
        toc7 = time.clock()

        collections.Counter(pred_test)
        # gt_test = gt[test_indices] - 1
        gt_test = gt1[test_indices] - 1
        overall_acc_mss = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
        oa_all[0][num] = overall_acc_mss
        confusion_matrix_mss = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
        each_acc_mss, average_acc_mss = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix_mss)
        aa_all[0][num] = average_acc_mss
        kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])
        kappa_all[0][num] = kappa

    KAPPA_RES_SS4.append(kappa)
    OA_RES_SS4.append(overall_acc_mss)
    AA_RES_SS4.append(average_acc_mss)

    print("training finished.")
    # print('Training Time: ', toc6 - tic6)
    # print('Test time:', toc7 - tic7)
    print("# %d Iteration" % (index_iter + 1))
    print('each_acc', each_acc_mss)
    print("oa", overall_acc_mss)
    print("aa", average_acc_mss)
    print("kappa", kappa)
    gt1[test_indices[:-VAL_SIZE]] = pred_test + 1
    gt1 = gt1.reshape(349, 1905)
    save_rgb('houston-DBDA.jpg', gt1, colors=spy_colors)
    #
    color = np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 1], [0.65, 0.35, 1],
         [0.75, 0.5, 0.75], [0.75, 1, 0.5], [0.5, 1, 0.65], [0.65, 0.65, 0], [0.75, 1, 0.65], [0, 0, 0.5], [0, 1, 0.75]])
    # color = color*255
    newcmap = ListedColormap(color)

    view = pyplot.imshow(gt1.astype(int), cmap=newcmap)
    bar = pyplot.colorbar()
    bar.set_ticks(np.linspace(0, 15, 16))

    pyplot.show()
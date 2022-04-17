# -*- coding: utf-8 -*-
# 整个模型训练测试验证代码，并保存最优模型，打印测试数据
import numpy as np
import scipy.io as sio
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
import time
import collections
from sklearn import metrics, preprocessing
from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy,deformable_se_sep_ksc

import matplotlib.pyplot as plt
import keras


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 产生新数据集的过程
# indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[result], PATCH_LENGTH)
def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        # counter 是从0开始计数的，是具体的值
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


# divide dataset into train and test datasets
def sampling(proptionVal, groundTruth):
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    print(m)
    # 16
    # 16类，对每一类样本要先打乱，然后再按比例分配，得到一个字典，因为上面是枚举，所以样本和标签的对应
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        # print(indices)
        # 每一类的样本数
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    # 将所有的训练样本存到train集合中，将所有的测试样本存到test集合中
    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    print(len(test_indices))
    # 27061
    print(len(train_indices))
    # 27068
    return train_indices, test_indices


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


# 调用设计好的模型
def model_MSDNet():
    model_dense = deformable_se_sep_ksc.ResnetBuilder.build_resnet_8((1, img_rows, img_cols, img_channels),
                                                       nb_classes)

    RMS = RMSprop(lr=0.0003)
    # Let's train the model using RMSprop
    model_dense.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])

    return model_dense


# 加载数据
mat_data = sio.loadmat('F:/transfer code/Tensorflow  Learning/SKNet/datasets/ksc/KSC.mat')
data_IN = mat_data['KSC']
# 标签数据
mat_gt = sio.loadmat('F:/transfer code/Tensorflow  Learning/SKNet/datasets/ksc/KSC_gt.mat')
gt_IN = mat_gt['KSC_gt']
# print('data_IN:',data_IN)
print(data_IN.shape)
# (512,614,176)
print(gt_IN.shape)
# (512,614)

# new_gt_IN = set_zeros(gt_IN, [result,4,7,9,13,15,16])
new_gt_IN = gt_IN

batch_size = 16

nb_classes = 13
nb_epoch = 200  # 400
img_rows, img_cols = 21, 21  # 27, 27
patience = 200

INPUT_DIMENSION_CONV = 176
INPUT_DIMENSION = 176

# 20%:10%:70% data for training, validation and testing

TOTAL_SIZE = 5211
VAL_SIZE = 521

TRAIN_SIZE = 3132
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.4  # 20% for trainnig and 80% for validation and testing
# 0.9  5212
# 0.8  1048
# 0.7  156360
# 0.6  2090
# 0.5
# 0.4  3132
# 0.3  3654

img_channels = 176
PATCH_LENGTH = 10  # Patch_size (13*2+result)*(13*2+result)

print(data_IN.shape[:2])
# (512,614)
print(np.prod(data_IN.shape[:2]))
# 314368
print(data_IN.shape[2:])
# (176,)
print(np.prod(data_IN.shape[2:]))
# 176
print(np.prod(new_gt_IN.shape[:2]))
# 314368

# 对数据进行reshape处理之后，进行scale操作
data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

# 标准化操作，即将所有数据沿行沿列均归一化道0-1之间
data = preprocessing.scale(data)
print(data.shape)
# (314368,176)

# 对数据边缘进行填充操作，有点类似之前的镜像操作
data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
whole_data = data_
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)
print(padded_data.shape)
# (524,626,176)
# 因为选择的是7*7的滑动窗口，145*145,145/7余5，也就是说有5个像素点扫描不到，所有在长宽每边个填充3，也就是6，这样的话
# 就可以将所有像素点扫描到

ITER = 1
CATEGORY = 13

train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
print(train_data.shape)
# (654,13,13,145)
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
print(test_data.shape)
# (2594,13,13,145)

# 评价指标
KAPPA_3D_MSDNet = []
OA_3D_MSDNet = []
AA_3D_MSDNet = []
TRAINING_TIME_3D_MSDNet = []
TESTING_TIME_3D_MSDNet = []
ELEMENT_ACC_3D_MSDNet = np.zeros((ITER, CATEGORY))

# seeds = [1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229]
seeds = [1334]

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))
    # # result Iteration

    # save the best validated model
    best_weights_MSDNet_path = 'F:/transfer code/Tensorflow  Learning/SKNet/models-ksc-21-613/ksc_best_3D_MSDNet_' + str(
        index_iter + 1) + '.hdf5'

    # 通过sampling函数拿到测试和训练样本
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    print('56' * 10, len(train_indices), len(test_indices))
    # train_indices 1048     test_indices 4163

    # gt本身是标签类，从标签类中取出相应的标签 -result，转成one-hot形式
    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    # 这个地方论文也解释了一下，是新建了一个以采集中心为主的新数据集，还是对元数据集进行了一些更改
    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    # 拿到了新的数据集进行reshpae之后，数据处理就结束了
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)

    # 在测试数据集上进行验证和测试的划分
    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    model_MSDNet = model_MSDNet()

    # 创建一个实例history
    history = LossHistory()

    # monitor：监视数据接口，此处是val_loss,patience是在多少步可以容忍没有提高变化
    earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
    # 用户每次epoch最后都会保存模型，如果save_best_only=True,那么最近验证误差最后的数据将会被保存下来
    saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_MSDNet_path, monitor='val_loss', verbose=1,
                                                save_best_only=True,
                                                mode='auto')

    # 训练和验证
    tic6 = time.clock()
    print(x_train.shape, x_test.shape)
    # (2055,7,7,200)  (7169,7,7,200)
    history_3d_MSDNet = model_MSDNet.fit(
        x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1), y_train,
        validation_data=(x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3], 1), y_val),
        batch_size=batch_size,
        nb_epoch=nb_epoch, shuffle=True, callbacks=[earlyStopping6, saveBestModel6, history])
    toc6 = time.clock()

    # 测试
    tic7 = time.clock()
    loss_and_metrics = model_MSDNet.evaluate(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1), y_test,
        batch_size=batch_size)
    toc7 = time.clock()

    print('3D MSDNet Time: ', toc6 - tic6)
    print('3D MSDNet Test time:', toc7 - tic7)

    print('3D MSDNet Test score:', loss_and_metrics[0])
    print('3D MSDNet Test accuracy:', loss_and_metrics[1])

    print(history_3d_MSDNet.history.keys())

    # 预测
    pred_test = model_MSDNet.predict(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)
    # 跟踪值出现的次数
    collections.Counter(pred_test)

    gt_test = gt[test_indices] - 1
    # print(len(gt_test))
    # 8194
    # 这是测试集，验证和测试还没有分开
    overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])
    KAPPA_3D_MSDNet.append(kappa)
    OA_3D_MSDNet.append(overall_acc)
    AA_3D_MSDNet.append(average_acc)
    TRAINING_TIME_3D_MSDNet.append(toc6 - tic6)
    TESTING_TIME_3D_MSDNet.append(toc7 - tic7)
    ELEMENT_ACC_3D_MSDNet[index_iter, :] = each_acc

    # 绘制acc-loss曲线
    history.loss_plot('epoch')

    print("3D MSDNet finished.")
    print("# %d Iteration" % (index_iter + 1))

# 自定义输出类
modelStatsRecord.outputStats(KAPPA_3D_MSDNet, OA_3D_MSDNet, AA_3D_MSDNet, ELEMENT_ACC_3D_MSDNet,
                             TRAINING_TIME_3D_MSDNet, TESTING_TIME_3D_MSDNet,
                             history_3d_MSDNet, loss_and_metrics, CATEGORY,
                             'F:/transfer code/Tensorflow  Learning/SKNet/records-ksc-21-613/IN_train_3D.txt',
                             'F:/transfer code/Tensorflow  Learning/SKNet/records-ksc-21-613/IN_train_3D_element.txt')

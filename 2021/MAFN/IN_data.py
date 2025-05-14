from pylab import *
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.io as sio
import keras
import numpy as np
import matplotlib.pyplot as pyplot
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn import metrics, preprocessing
from keras.models import load_model
import keras.callbacks as kcallbacks
import collections
from Utils import normalization, doPCA, modelStatsRecord, averageAccuracy, zeroPadding, averageAccuracy, Kappa
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
from keras.callbacks import ReduceLROnPlateau
import time
import IN_net
from spectral import spy_colors, save_rgb
# import spectral
np.random.seed(1337)

def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0

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
    m = max(groundTruth)

    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)

        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        print('class :', i, 'num :', nb_val, 'len :', int(len(indices)))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
#   whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
#        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices
    print(len(test_indices))


mat_data = sio.loadmat('dataset/IN/Indian_pines_corrected.mat')
print(mat_data.keys())
data_IN = mat_data['indian_pines_corrected']
print(data_IN.shape)

mat_data1 = sio.loadmat('dataset/IN/IN_PCA_30.mat')
print(mat_data.keys())
data_IN1 = mat_data1['data']
print(data_IN1.shape)


mat_gt = sio.loadmat('dataset/IN/Indian_pines_gt.mat')
print(mat_gt.keys())
gt_IN = mat_gt['indian_pines_gt']
print(gt_IN.shape)


new_gt_IN = gt_IN

nb_classes = 16
batch_size = 16
nb_epoch = 200
patience = 200

img_rows, img_cols = 7, 7
img_rows1, img_cols1 = 27, 27


INPUT_DIMENSION_CONV = 200
INPUT_DIMENSION_CONV1 = 30
img_channels = 200
img_channels1 = 3

PATCH_LENGTH = 3
PATCH_LENGTH1 = 13

TOTAL_SIZE = 10249
VAL_SIZE =1025  # 10%
TRAIN_SIZE = 520  # 5%
# TRAIN_SIZE = 419  # 4%
# TRAIN_SIZE = 314  # 3%
# TRAIN_SIZE = 212  # 2%
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
ALL_SIZE = data_IN.shape[0] * data_IN.shape[1]
VALIDATION_SPLIT = 0.95

data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
print('data.shape：', data.shape)
data1 = data_IN1.reshape(np.prod(data_IN1.shape[:2]), np.prod(data_IN1.shape[2:]))
print('data.shape：', data1.shape)
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)
print('gt.shape:', gt.shape)

data = preprocessing.scale(data)
data_ = data.reshape(data_IN.shape[0], data_IN.shape[1],data_IN.shape[2])

data1 = preprocessing.scale(data1)
data1_ = data1.reshape(data_IN1.shape[0], data_IN1.shape[1],data_IN1.shape[2])

whole_data = data_
whole_data1 = data1_

padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)
print("whole_data.shape:", whole_data.shape)

padded_data1 = zeroPadding.zeroPadding_3D(whole_data1, PATCH_LENGTH1)
print("whole_data.shape:", whole_data1.shape)


ITER = 1
CATEGORY = 16

train_data = np.zeros((TRAIN_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
train_data1 = np.zeros((TRAIN_SIZE, 2*PATCH_LENGTH1 + 1, 2*PATCH_LENGTH1 + 1, INPUT_DIMENSION_CONV1))


test_data = np.zeros((TEST_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
test_data1 = np.zeros((TEST_SIZE, 2*PATCH_LENGTH1 + 1, 2*PATCH_LENGTH1 + 1, INPUT_DIMENSION_CONV1))

seeds=[1220]

# best_weights_RES_path_ss4 = 'models/Indian_best1.hdf5'

NUM=1
for num in range(NUM):
    for index_iter in range(ITER):
        print("# %d Iteration" % (index_iter + 1))

        # best_weights_RES_path_ss = 'models/two_cha/IN5%/In_' + str(
        #     index_iter + 1) + '.hdf5'
        np.random.seed(seeds[index_iter])

        train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
        m = len(test_indices)
        n = len(train_indices)
        print('test', m)
        print('train', n)

        y_train = gt[train_indices] - 1
        y_train = to_categorical(np.asarray(y_train))

        y_test = gt[test_indices] - 1
        gt_test = y_test[:-VAL_SIZE]
        y_test = to_categorical(np.asarray(y_test))

        train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
        train_assign1 = indexToAssignment(train_indices, whole_data1.shape[0], whole_data1.shape[1], PATCH_LENGTH1)

        # np.random.seed(None)
        for i in range(len(train_assign)):
            train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

            train_data1[i] = selectNeighboringPatch(padded_data1, train_assign1[i][0], train_assign1[i][1], PATCH_LENGTH1)


        test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
        test_assign1 = indexToAssignment(test_indices, whole_data1.shape[0], whole_data1.shape[1], PATCH_LENGTH1)

        # sess2=tf.Session()
        for i in range(len(test_assign)):
            test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)
            test_data1[i] = selectNeighboringPatch(padded_data1, test_assign1[i][0], test_assign1[i][1], PATCH_LENGTH1)

        all_assign = indexToAssignment(range(ALL_SIZE), whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
        all_assign1 = indexToAssignment(range(ALL_SIZE), whole_data1.shape[0], whole_data1.shape[1], PATCH_LENGTH1)

        x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2],
                                     INPUT_DIMENSION_CONV)
        x_train1 = train_data1.reshape(train_data1.shape[0], train_data1.shape[1], train_data1.shape[2],
                                       INPUT_DIMENSION_CONV1)

        x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2],
                                       INPUT_DIMENSION_CONV)
        x_test_all1 = test_data1.reshape(test_data1.shape[0], test_data1.shape[1], test_data1.shape[2],
                                         INPUT_DIMENSION_CONV1)

        x_val = x_test_all[-VAL_SIZE:]
        x_val1 = x_test_all1[-VAL_SIZE:]
        y_val = y_test[-VAL_SIZE:]

        x_test = x_test_all[:-VAL_SIZE]
        x_test1 = x_test_all1[:-VAL_SIZE]
        y_test = y_test[:-VAL_SIZE]

        print("x_train shape :", x_train.shape, x_train1.shape)
        print("y_train shape :", y_train.shape)
        print('x_val shape :', x_val.shape, x_val1.shape)
        print('y_val shape :', y_val.shape)
        print("x_test shape :", x_test.shape, x_test1.shape)
        print("y_test shape :", y_test.shape)


model = IN_net.model()

earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
saveBestModel6 = kcallbacks.ModelCheckpoint(filepath='ckpt/indian_pines.h5', monitor='val_loss', verbose=1,save_best_only=True, mode='auto')

print("^-^-------------training-------------------^-^")

tic6 = time.clock()
History = model.fit([x_train, x_train1],  y_train, nb_epoch =nb_epoch, batch_size=16,
          callbacks=[earlyStopping6, saveBestModel6], validation_data=([x_val, x_val1],  y_val))
toc6 = time.clock()
model.save("models/IN.h5")

scores = model.evaluate([x_val, x_val1],  y_val, batch_size=25)
print('Test score:', scores[0])
print('Test accuracy:', scores[1])
model.save_weights('model_weigh/IN.h5')


print("^-^-------------testing-------------------^-^")
model.load_weights('ckpt/indian_pines.h5')
tic7 = time.clock()
pred_test = model.predict([x_test, x_test1]).argmax(axis=1)
toc7 = time.clock()


collections.Counter(pred_test)
gt_test = gt[test_indices] - 1
gt_train = gt[train_indices] - 1
overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])

confusion_matrix_res4 = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
each_acc_res4, average_acc_res4 = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix_res4)

kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])

collections.Counter(pred_test)
gt_test = gt[test_indices] - 1
gt_train = gt[train_indices] - 1

overaccy=metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
confusion_matrix_mss=metrics.confusion_matrix(gt_test[:-VAL_SIZE],pred_test)
print(confusion_matrix_mss)

average_acc=averageAccuracy.AA_andEachClassAccuracy(confusion_matrix_mss)
kappa_value=Kappa.kappa(confusion_matrix_mss)


print("training finished.")
print('Training Time: ', toc6 - tic6)
print('Test time:', toc7 - tic7)

print('each_acc', each_acc_res4)
print("aa", average_acc_res4)
print("oa", overall_acc)
print("kappa", kappa)

gt[test_indices[:-VAL_SIZE]] = pred_test + 1
gt = gt.reshape(145, 145)
save_rgb('IN-DBDA.jpg', gt, colors=spy_colors)
#
color = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 1], [0.65, 0.35, 1],
                  [0.75, 0.5, 0.75], [0.75, 1, 0.5], [0.5, 1, 0.65], [0.65, 0.65, 0], [0.75, 1, 0.65], [0, 0, 0.5], [0, 1, 0.75], [0.5, 0.75, 1]])
# color = color*255
newcmap = ListedColormap(color)

view = pyplot.imshow(gt.astype(int), cmap=newcmap)
bar = pyplot.colorbar()
bar.set_ticks(np.linspace(0, 16, 17))
bar.set_ticklabels(('', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-tree', 'Grass-pasture-mowed', 'Hay-windrowed',
                    'Oats', 'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'))
pyplot.show()
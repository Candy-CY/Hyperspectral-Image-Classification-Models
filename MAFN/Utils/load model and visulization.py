from keras.layers import Dense, Input
from keras.models import Model
import scipy.io as sio
import numpy as np
import time
import h5py
import matplotlib.pyplot as plt

def create_model():
    input_pixel = Input(shape=(103,))
    encoded = Dense(60, activation='sigmoid', init='he_normal')(input_pixel)
    # encoded = BatchNormalization()(encoded)
    encoded = Dense(60, activation='sigmoid', init='he_normal')(encoded)
    # encoded = BatchNormalization()(encoded)
    encoded = Dense(60, activation='sigmoid', init='he_normal')(encoded)
    # encoded = BatchNormalization()(encoded)
    encoded = Dense(60, activation='sigmoid', init='he_normal')(encoded)
    # encoded = BatchNormalization()(encoded)
    # encoded = Dropout(0.5)(encoded)
    preds = Dense(9, activation='softmax')(encoded)

    model = Model(input_pixel, preds)
    return model

def load_trained_model(weight_path):
    model = create_model()
    model.load_weights(weight_path)
    return model

best_weights_path = '/home/finoa/DL-on-HSI-Classification/Best_models/best_1D_AE_BN.hdf5'
uPavia = sio.loadmat('/home/finoa/DL-on-HSI-Classification/Datasets/UPavia/PaviaU.mat')
gt_uPavia = sio.loadmat('/home/finoa/DL-on-HSI-Classification/Datasets/UPavia/PaviaU_gt.mat')
data_UP = uPavia['paviaU']
gt_UP = gt_uPavia['paviaU_gt']
data = data_UP.reshape(np.prod(data_UP.shape[:2]),np.prod(data_UP.shape[2:]))
gt = gt_UP.reshape(np.prod(gt_UP.shape[:2]),)

MEAN = np.mean(data, axis=0)
MAX = np.max(data.ravel())

weight_path = '/home/finoa/DL-on-HSI-Classification/Best_models/best_1D_AE_BN.hdf5'
model = load_trained_model(weight_path)

input_data = data - MEAN
input_data /= MAX

result_image = model.predict(input_data)

result = []
for i in range(result_image.shape[0]):
    vec = result_image[i].ravel().tolist()
    result.append(vec.index(max(vec)))


x = np.ravel(gt)
# print x
y = np.zeros((x.shape[0], 3))
z = np.zeros(x.shape)

for index, item in enumerate(x):
    if item == 0:
        y[index] = np.array([0, 0, 0]) / 255.
    if item == 1:
        y[index] = np.array([128, 128, 128]) / 255.
    if item == 2:
        y[index] = np.array([128, 0, 0]) / 255.
    if item == 3:
        y[index] = np.array([128, 0, 128]) / 255.
    if item == 4:
        y[index] = np.array([0, 255, 255]) / 255.
    if item == 5:
        y[index] = np.array([0, 255, 255]) / 255.
    if item == 6:
        y[index] = np.array([255, 0, 255]) / 255.
    if item == 7:
        y[index] = np.array([255, 0, 0]) / 255.
    if item == 8:
        y[index] = np.array([255, 255, 0]) / 255.
    if item == 9:
        y[index] = np.array([0, 128, 0]) / 255.

for index, item in enumerate(x):
    if item == 0:
        z[index] = 0
    else:
        z[index] = 1
        # print y

plt.figure(1)
plt.title('Ground truth')
y_re = np.reshape(y, (gt_UP.shape[0], gt_UP.shape[1], 3))
print y_re
plt.imshow(y_re)
plt.show()

result_arr = np.asarray(result) + 1

result_arr = np.asarray(result_arr) * np.asarray(z)           #multiply mask

x = np.ravel(result_arr.tolist())
y = np.zeros((x.shape[0], 3))

for index, item in enumerate(x):
    if item == 0:
        y[index] = np.array([0,0,0])/255.
    if item == 1:
        y[index] = np.array([128,128,128])/255.
    if item == 2:
        y[index] = np.array([128,0,0])/255.
    if item == 3:
        y[index] = np.array([128,0,128])/255.
    if item == 4:
        y[index] = np.array([0,255,255])/255.
    if item == 5:
        y[index] = np.array([0,255,255])/255.
    if item == 6:
        y[index] = np.array([255,0,255])/255.
    if item == 7:
        y[index] = np.array([255,0,0])/255.
    if item == 8:
        y[index] = np.array([255,255,0])/255.
    if item == 9:
        y[index] = np.array([0,128,0])/255.

plt.figure(2)
plt.title('Classificaion result')
y_re = np.reshape(y,(gt_UP.shape[0],gt_UP.shape[1],3))
print y_re
plt.imshow(y_re)
plt.show()

from sklearn import preprocessing
import numpy as np
# import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import torch.utils.data as Data
import torch.nn as nn
import random
from torchvision.transforms import transforms
# from sklearn.decomposition import PCA


def data_processing(Dataset='UP', batch_size=64, PATCH_LENGTH=11, flag='Train', Train_size=76, noise_type='sym', noise_ratio=24):
    # PATCH_LENGTH is the total length
    print('The name of Dataset:', Dataset)
    data_hsi, gt, TOTAL_SIZE = load_dataset(Dataset)

    print(data_hsi.shape)
    image_x, image_y, BAND = data_hsi.shape
    data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))  # h*w row, d column
    gt_h = gt.reshape(np.prod(gt.shape[:2]), )
    CLASSES_NUM = max(gt_h)
    print('The class numbers of the HSI data is:', CLASSES_NUM)


    INPUT_DIMENSION = BAND
    data = preprocessing.scale(data)

    data = data.reshape(data_hsi.shape[0], data_hsi.shape[1], INPUT_DIMENSION)
    padded_data = np.lib.pad(data,
                             ((PATCH_LENGTH // 2, PATCH_LENGTH // 2), (PATCH_LENGTH // 2, PATCH_LENGTH // 2), (0, 0)),
                             'constant', constant_values=0)  # padding

    if flag == 'testAll':
        true_point = get_all_index(gt, CLASSES_NUM)
        train_iter = generate_all(true_point, PATCH_LENGTH, padded_data, INPUT_DIMENSION)
        return train_iter, INPUT_DIMENSION, CLASSES_NUM, data_hsi.shape
    if flag == 'train':
        train_indices, train_gt, test_indices, test_gt = sampling(Train_size, gt, CLASSES_NUM)
        train_iter, valid_iter, test_iter = generate_iter(train_gt, train_indices, test_gt, test_indices, train_indices.shape[0],
                     PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, noise_type, noise_ratio)
        return train_iter, valid_iter, test_iter


class RandomSpectralShift(nn.Module):
    def __init__(self, shift_spectral_num=10):
        super(RandomSpectralShift, self).__init__()
        self.shift_spectral_num = shift_spectral_num
        # self.channels_range = list(range(spectral_num))  # feature_channels

    def forward(self, x):
        # print(x.shape)
        x = self.shift(x, shift_spectral_num=self.shift_spectral_num, channels_range=list(range(x.shape[2])))
        return x

    @staticmethod
    def shift(x, shift_spectral_num, channels_range):
        # print(x.shape)
        # x = rearrange(x, 'C S H W -> C H W S')
        # print(x.shape)
        all = random.sample(channels_range, shift_spectral_num)
        forward = sorted(all[: shift_spectral_num // 2])
        backward = sorted(all[shift_spectral_num // 2:])
        fixed = list(set(channels_range) - set(all))

        out = np.zeros_like(x)
        # print(out.shape)
        out[:, : -1, forward] = x[:, 1:, forward]  # shift left
        out[:, 1:, backward] = x[:, :-1, backward]  # shift right
        out[:-1, :, forward] = x[1:, :, forward]  # shift left
        out[1:, :, backward] = x[:-1, :, backward]  # shift right
        out[:, :, fixed] = x[:, :, fixed]  # not shift

        # out = rearrange(out, 'H W S -> S H W')
        return out


class RandomSpatialRotate(nn.Module):
    def __init__(self, p=0.05):
        super(RandomSpatialRotate, self).__init__()
        self.p = p

    def forward(self, x):
        if round(np.random.uniform(0, 1), 1) <= self.p:
            x = self.rotate(x)
        return x

    def rotate(self, x):
        idx = random.randint(0, 3)
        out = np.rot90(x, k=idx, axes=(0, 1)).copy()
        return out


class HSIDA(Data.Dataset):
    def __init__(
            self,
            images,
            labels,
            mode,
    ) -> None:

        self.transform_train_weak = transforms.Compose([
            RandomSpatialRotate(),
        ])
        self.transform_train_strong = transforms.Compose([
            RandomSpectralShift(),
            RandomSpatialRotate(),
        ])
        self.transform_test = transforms.Compose([

        ])

        self.mode = mode
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image, target = self.images[index], self.labels[index]
        # image.shape = 9,9,200
        if self.mode == 'preTrain':
            img = torch.from_numpy(image.transpose(2,0,1)).type(torch.FloatTensor)
            return img, target
        elif self.mode == 'train':
            raw = torch.from_numpy(self.transform_train_weak(image).transpose(2,0,1)).type(torch.FloatTensor)
            img1 = torch.from_numpy(self.transform_train_strong(image).transpose(2,0,1)).type(torch.FloatTensor)
            img2 = torch.from_numpy(self.transform_train_strong(image).transpose(2,0,1)).type(torch.FloatTensor)
            return raw, img1, img2, target
        elif self.mode == 'test':
            img = torch.from_numpy(image.transpose(2,0,1)).type(torch.FloatTensor)
            return img, target

    def __len__(self):
        return len(self.labels)


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


def train_and_test_data(mirror_image, band, train_point, test_point, patch=5):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=np.float32)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=np.float32)

    # patch
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)

    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))  # num,patch,patch,bands
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    # print("x_true  shape = {}, type = {}".format(x_true.shape, x_test.dtype))
    print("**************************************************")

    return x_train, x_test


def generate_iter(label_train, train_indices, y_test, test_indices, VAL_SIZE,
                  PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, noise_type, noise_ratio):

    x_train, x_test_all = train_and_test_data(padded_data, INPUT_DIMENSION, train_indices, test_indices,
                                                patch=PATCH_LENGTH)
    y_test = np.array(y_test)
    y_train = get_noisy_label(label_train, percent=noise_ratio, noise_type=noise_type)

    indices = np.random.permutation(len(y_test))

    x_val = x_test_all[indices][-VAL_SIZE:]
    y_val = y_test[indices][-VAL_SIZE:]
    x_test = x_test_all[indices][:-VAL_SIZE]
    y_test = y_test[indices][:-VAL_SIZE]
    # print(' y_test shape:', y_test.shape)
    # print('Noisy label:', y_train)
    noise_rate = np.subtract(np.array(y_train), np.array(label_train))
    noise_ratio = len(noise_rate[np.nonzero(noise_rate)]) / float(len(y_train))
    print('True_noise_ratio:', noise_ratio)

    torch_dataset_train = HSIDA(images=x_train, labels=torch.from_numpy(y_train).type(torch.LongTensor), mode='train')

    torch_dataset_valida = HSIDA(images=x_val, labels=torch.from_numpy(y_val).type(torch.LongTensor), mode='test')

    torch_dataset_test = HSIDA(images=x_test, labels=torch.from_numpy(y_test).type(torch.LongTensor), mode='test')

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )
    valiada_iter = Data.DataLoader(
        dataset=torch_dataset_valida,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,
        num_workers=0,
    )
    return train_iter, valiada_iter, test_iter


def get_noisy_label(tru_label, percent, noise_type='sym'):
    tru_label = np.asarray(tru_label)
    y_train = np.array(tru_label)
    start = int(min(tru_label))
    end = int(max(tru_label))
    if noise_type == 'sym':
        if percent < 1:
            indices = np.random.permutation(len(y_train))
            for i, idx in enumerate(indices):
                if i < percent * len(y_train):
                    temp_list = np.arange(start, end + 1).tolist()
                    temp_list.remove(y_train[idx])
                    y_train[idx] = np.random.choice(temp_list)
                    # y_train[idx] = np.random.randint(low=start, high=end + 1, dtype=np.int32)
        elif percent > 1:
            # random disruption
            for i in range(start, end + 1):
                ttemp = np.where(tru_label == i)
                # print('index label ',ttemp)
                temp_list = np.arange(start, end + 1).tolist()
                temp_list.remove(i)  # 创建一个没有当前类别的list范围  然后下面的np.random.choice从中随机选取
                number = len(y_train[ttemp])
                # print('number of class', number)
                # print('before:',y_train[np.where(y_train==i)])
                temp = np.random.choice(temp_list, size=number)
                temp[percent:] = i
                # print(temp)
                np.random.shuffle(temp)
                # print(temp)
                y_train[ttemp] = temp
                # print('after:', y_train[ttemp])

    elif noise_type == 'asym':
        indices = np.random.permutation(len(y_train))
        if percent < 1:
            for i, idx in enumerate(indices):
                if i < percent * len(y_train):
                    y_train[idx] = (y_train[idx] + 1) % (max(tru_label) + 1) + min(tru_label)
        elif percent > 1:
            for i in range(start, end + 1):
                ttemp = np.where(tru_label == i)
                temp_value = (i + 1) % (end + 1) + start  # to its next
                number = len(y_train[ttemp])
                # print(number)
                # print('before:',y_train[np.where(y_train==i)])
                temp = np.ones(number) * temp_value
                temp[percent:] = i
                print(temp)
                np.random.shuffle(temp)
                # print(temp)
                y_train[ttemp] = temp
    return y_train


def generate_all(true_point, patch, mirror_image, band, batch_size=128):
    gt_all = np.zeros(true_point.shape[0])
    # print(gt.shape)

    all_data = np.zeros((true_point.shape[0], patch, patch, band), dtype=np.float32)
    for k in range(true_point.shape[0]):
        all_data[k, :, :, :] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)

    print("All data: shape = {}, type = {}".format(all_data.shape, all_data.dtype))
    torch_dataset_all = HSIDA(images=all_data, labels=torch.from_numpy(gt_all).type(torch.LongTensor), mode='test')

    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # attention! False is required for full map
        num_workers=0,
    )
    # return all_iter  # , y_test
    return all_iter


def load_dataset(Dataset):
    if Dataset == 'UP':
        uPavia = sio.loadmat('../dataset/PaviaU.mat')
        gt_uPavia = sio.loadmat('../dataset/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776

    if Dataset == 'SV':
        SV = sio.loadmat('../dataset/Salinas_corrected.mat')
        gt_SV = sio.loadmat('../dataset/Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129

    if Dataset == 'KSC':
        KSC = sio.loadmat('../dataset/KSC.mat')
        gt_KSC = sio.loadmat('../dataset/KSC_gt.mat')
        data_hsi = KSC['KSC']
        gt_hsi = gt_KSC['KSC_gt']
        TOTAL_SIZE = 5211

    return data_hsi, gt_hsi, TOTAL_SIZE


def sampling(proportion, ground_truth, num_class):
    # proportion : percentage for training
    train = {}
    test = {}
    train_gt = []
    test_gt = []
    for i in range(num_class):
        # indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        each_class = np.argwhere(ground_truth == i + 1)  #
        number = each_class.shape[0]
        tmp = np.ones(number)*i
        np.random.shuffle(each_class)
        # labels_loc[i] = indexes
        if proportion < 1:
            nb_tra = int(np.ceil(proportion * number))
            # temp = np.random.randint(low=0, high=number, size=nb_val)
        if proportion > 1:
            nb_tra = proportion
            if proportion > number:
                nb_tra = 15
            # temp = np.random.randint(low=0, high=number, size=nb_val)
            # print(nb_val)
        train[i] = each_class[:nb_tra, :]
        test[i] = each_class[nb_tra:, :]
        train_gt += tmp[:nb_tra].tolist()
        # print(train_gt)
        test_gt += tmp[nb_tra:].tolist()

    train_pos = train[0]
    for i in range(1, num_class):
        train_pos = np.r_[train_pos, train[i]]
    train_pos = train_pos.astype(int)

    test_pos = test[0]
    for i in range(1, num_class):
        test_pos = np.r_[test_pos, test[i]]
    test_pos = test_pos.astype(int)

    print("The num of labeled data for noisy training:", train_pos.shape[0])
    print("The num of labeled data for test and validation:", test_pos.shape[0])
    print("The total num of labeled data ", train_pos.shape[0] + test_pos.shape[0])

    return train_pos, train_gt, test_pos, test_gt


def get_all_index(true_data, num_classes):
    total_pos =[]
    for i in range(true_data.shape[0]):
        for j in range(true_data.shape[1]):
            total_pos.append([i, j])
    total_pos = np.array(total_pos)
    # print(total_pos)

    print("The num of all data for drawing the full classification map:", len(total_pos))

    return total_pos


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

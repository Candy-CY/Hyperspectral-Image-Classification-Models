# -*- coding:utf-8 -*-
# @Time       :2022/5/19 下午4:11
# @FileName   :utils.py
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import numpy as np
import random

import auxil
from hyper_pytorch import HyperData
from sklearn.model_selection import train_test_split


def normalize(input2):
    input2_normalize = np.zeros(input2.shape)
    for i in range(input2.shape[2]):
        input2_max = np.max(input2[:, :, i])
        input2_min = np.min(input2[:, :, i])
        input2_normalize[:, :, i] = (input2[:, :, i] - input2_min) / (input2_max - input2_min)

    return input2_normalize


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False


def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]


def split_data(pixels, labels, percent, splitdset="custom", rand_state=345):
    splitdset = "sklearn"
    if splitdset == "sklearn":
        return train_test_split(pixels, labels, test_size=(1 - percent), stratify=labels, random_state=rand_state)
    elif splitdset == "custom":
        pixels_number = np.unique(labels, return_counts=1)[1]
        train_set_size = [int(np.ceil(a * percent)) for a in pixels_number]
        tr_size = int(sum(train_set_size))
        te_size = int(sum(pixels_number)) - int(sum(train_set_size))
        sizetr = np.array([tr_size] + list(pixels.shape)[1:])
        sizete = np.array([te_size] + list(pixels.shape)[1:])
        train_x = np.empty((sizetr))
        train_y = np.empty((tr_size))
        test_x = np.empty((sizete))
        test_y = np.empty((te_size))
        trcont = 0
        tecont = 0
        for cl in np.unique(labels):
            pixels_cl = pixels[labels == cl]
            labels_cl = labels[labels == cl]
            pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
            for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
                if cont < train_set_size[cl]:
                    train_x[trcont, :, :, :] = a
                    train_y[trcont] = b
                    trcont += 1
                else:
                    test_x[tecont, :, :, :] = a
                    test_y[tecont] = b
                    tecont += 1
        train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
        return train_x, test_x, train_y, test_y


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes + 1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)

    return y_train, y_test, y_true


def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=np.float32)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=np.float32)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=np.float16)
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k, :, :, :] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)

    return x_train, x_test, x_true


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


def choose_train_test_sample(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}

    # -------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data == (i + 1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]  # (695,2)
    total_pos_train = total_pos_train.astype(int)
    # --------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    total_pos_test = total_pos_test.astype(int)
    # --------------------------for true data------------------------------------
    for i in range(num_classes + 1):
        each_class = []
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes + 1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


def data_load(data, labels, spatialsize, val_percent, tr_bsize=64, te_bsize=1000, use_val=True):
    '''
    According to the number of training samples, the training, validation and test sets are randomly selected.
    :param data: HSI
    :param labels: GT
    :param spatialsize:
    :param val_percent:
    :param tr_bsize:
    :param te_bsize:
    :param use_val: Specifies whether to use a validation set.
    :return:
    '''
    kwargs = {'num_workers': 0, 'pin_memory': True}

    ####################################################################################################################
    # training labels with fixed number of GT
    h, w = labels.shape[0], labels.shape[1]
    labels = labels.reshape(h * w)
    num_class = np.max(labels)

    # houstonu
    train_num = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

    train_label = np.zeros_like(labels)
    test_label = np.zeros_like(labels)

    for i in range(num_class):
        r = random.random
        random.seed(0)
        index = np.where(labels == i + 1)[0]
        random.shuffle(index, r)
        train_index = index[:train_num[i]]
        test_index = index[train_num[i]:]
        train_label[train_index] = labels[train_index]
        test_label[test_index] = labels[test_index]

    labels = labels.reshape(h, w)
    train_label = train_label.reshape(h, w)
    test_label = test_label.reshape(h, w)

    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = choose_train_test_sample(
        train_label, test_label, labels, num_class)

    margin = int(spatialsize / 2)
    zeroPaddedX = padWithZeros(data, margin=margin)
    x_train, x_test, x_true = train_and_test_data(zeroPaddedX, band=zeroPaddedX.shape[-1],
                                                  train_point=total_pos_train, test_point=total_pos_test,
                                                  true_point=total_pos_true, patch=spatialsize)
    y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_class)

    del total_pos_train, total_pos_test, number_train, number_test, number_true, zeroPaddedX

    if use_val:
        x_val, x_test, y_val, y_test = split_data(x_test, y_test, val_percent)

    train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"), y_train))
    test_hyper = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"), y_test))

    if use_val:
        val_hyper = HyperData((np.transpose(x_val, (0, 3, 1, 2)).astype("float32"), y_val))
    else:
        val_hyper = None

    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=tr_bsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_hyper, batch_size=te_bsize, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_hyper, batch_size=te_bsize, shuffle=False, **kwargs)

    return train_loader, test_loader, val_loader


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def train_epoch(model, train_loader, criterion, optimizer, use_cuda=True):
    model.train()
    accs = np.ones((len(train_loader))) * -1000.0
    losses = np.ones((len(train_loader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs, con_loss = model(inputs)

        loss = (1 - 0.8) * criterion(outputs, targets) + 0.8 * con_loss

        losses[batch_idx] = loss.item()
        accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return (np.average(losses), np.average(accs))


def valid_epoch(model, valid_loader, criterion, use_cuda=True):
    model.eval()
    accs = np.ones((len(valid_loader))) * -1000.0
    losses = np.ones((len(valid_loader))) * -1000.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs, con_loss = model(inputs)
            loss = (1 - 0.8) * criterion(outputs, targets) + 0.8 * con_loss
            losses[batch_idx] = loss.item()
            accs[batch_idx] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return (np.average(losses), np.average(accs))


def predict(testloader, model, criterion, use_cuda):
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs = inputs.cuda()
            [predicted.append(F.softmax(a).cpu().numpy()) for a in model(inputs)[0].data]

    return np.array(predicted)


import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)

            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

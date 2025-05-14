import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA

import auxil
from model import NCGLF
from utils import train_epoch, valid_epoch, data_load, FocalLoss, predict
from sklearn.preprocessing import StandardScaler
import torch
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str('0')  # 指定显卡位置


def main(patchsize, epoches):

    # 导入数据
    hsi = loadmat('datasets/houston2013/HoustonU_hsi.mat')['hsi']  # (349,1905,144)
    lidar = loadmat('datasets/houston2013/HoustonU_lidar.mat')['lidar'][:, :, None]  # (349,1905)
    gt = loadmat('datasets/houston2013/HoustonU_gt.mat')['img_gt']  # (349,1905)

    hsi = hsi.astype(np.float32)
    lidar = lidar.astype(np.float32)
    gt = gt.astype(np.int8)

    height1, width1, band1 = hsi.shape
    height2, width2, band2 = lidar.shape

    print('Hyperspectral data size: ', hsi.shape)  # 打印HSI数据大小
    print('Lidar data size: ', lidar.shape)  # 打印Lidar数据大小

    dimension = 40  # 指定HSI维度

    newX = np.reshape(hsi, (-1, hsi.shape[2]))
    pca = PCA(n_components=dimension, whiten=True)
    newX = pca.fit_transform(newX)

    hsi = np.reshape(newX, (hsi.shape[0], hsi.shape[1], dimension))

    data = np.concatenate([hsi, lidar], axis=2)  # HSI与Lidar数据叠加

    [m1, n1, l1] = np.shape(data)
    data = data.reshape(m1 * n1, l1)
    data = StandardScaler().fit_transform(data)  # 标准化
    data = data.reshape(m1, n1, l1)

    num_classes = len(np.unique(gt)) - 1  # number of classes
    train_loader, test_loader, val_loader = data_load(data, gt, patchsize, val_percent=0.15, use_val=True)  # 筛选训练集、验证集和测试集

    # -------------------------------------------------------------------------------
    # create model
    model = NCGLF(num_classes=num_classes, patch_size=patchsize, encoder_dim=64, depth=2, c1=dimension, c2=band2).cuda()
    # criterion
    criterion = FocalLoss().cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoches // 5, gamma=0.9)

    best_acc = -1
    print("start training")
    tic1 = time.time()
    train_losses = []
    train_acces = []
    # 用数组保存每一轮迭代中，在测试数据上测试的损失值和精度值，为了通过画图展示出来
    eval_losses = []
    eval_acces = []
    for epoch in range(epoches):
        # train model
        model.train()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)
        train_acces.append(train_acc)

        scheduler.step()
        eval_loss, eval_acc = valid_epoch(model, val_loader, criterion)
        eval_losses.append(eval_loss)
        eval_acces.append(eval_acc)

        print('epoch: {} | train loss: {} train acc: {} | eval loss: {} eval acc: {}'
              .format(epoch, train_loss, train_acc, eval_loss, eval_acc))

        # save model
        if eval_acc > best_acc:
            best_acc = eval_acc
            state = {
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(state, "best_model.pth.tar")


    toc1 = time.time()
    checkpoint = torch.load("best_model.pth.tar")
    best_acc = checkpoint['best_acc']

    model.load_state_dict(checkpoint['state_dict'])

    print("FINAL:      ACCURACY", best_acc)
    tr_time = toc1 - tic1
    tic2 = time.time()
    prediction = predict(test_loader, model, criterion, True)
    prediction = np.argmax(prediction, axis=1)
    classification, confusion, results = auxil.reports(prediction, np.array(test_loader.dataset.__labels__()))
    toc2 = time.time()
    te_time = toc2 - tic2
    print("Final records:")

    print("Running Traing Time: {:.2f}".format(tr_time))
    print("Running Test Time: {:.2f}".format(te_time))
    print("**************************************************")
    return results[0], results[1], results[2], results[3:len(results)], tr_time, te_time


if __name__ == '__main__':
    num = 1
    OA = []
    AA = []
    KAPPA = []
    CA = []
    spatial_size = 11  # patch size
    epoches = 80  # 设置迭代次数
    Train_t = []
    Test_t = []

    for i in range(num):
        print('iteration {}'.format(i + 1))
        oa, aa, kappa, ca, tr_time, te_time = main(spatial_size, epoches)
        OA = np.hstack((OA, oa))
        AA = np.hstack((AA, aa))
        KAPPA = np.hstack((KAPPA, kappa))
        CA.append(ca)
        Train_t = np.hstack((Train_t, tr_time))
        Test_t = np.hstack((Test_t, te_time))

    OA_average = np.mean(OA)
    OA_std = np.std(OA)
    AA_average = np.mean(AA)
    AA_std = np.std(AA)
    KAPPA_average = np.mean(KAPPA)
    KAPPA_std = np.std(KAPPA)
    CA_average = np.mean(CA, axis=0)
    CA_std = np.std(CA, axis=0)
    Train_t_avg = np.mean(Train_t)
    Test_t_avg = np.mean(Test_t)

    print("OA avg: {:.4f}, AA avg: {:.4f}, Kappa avg: {:.4f}".format(OA_average, AA_average, KAPPA_average))
    print("OA std: {:.4f}, AA std: {:.4f}, Kappa std: {:.4f}".format(OA_std, AA_std, KAPPA_std))
    print("CA avg: {}".format(CA_average))
    print("CA std: {}".format(CA_std))
    print("Train_t avg: {}".format(Train_t_avg))
    print("Test_t avg: {}".format(Test_t_avg))

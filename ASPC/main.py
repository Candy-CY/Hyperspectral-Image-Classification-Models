# -*- coding: utf-8 -*-
"""
@author: Zhu
"""
import numpy as np
import torch
from func.Hyper import matDataLoad, split_dataset_equal, dataNormalize, sys_random_fixed
import cv2
import argparse

""" 设置系统参数及CPU """
parser = argparse.ArgumentParser(description='相关参数 可在命令行编辑')
parser.add_argument('--dataset', type=str, default='Salinas', help='数据集')
parser.add_argument('--numTrain', type=int, default=20, help='每类训练样本数量')
parser.add_argument('--lr', type=float, default=0.08, help='学习率')
parser.add_argument('--lr_dec_epoch', type=int, default=100, help='每N轮衰减一次学习率')
parser.add_argument('--lr_dec_rate', type=float, default=0.9, help='学习率衰减率')
parser.add_argument('--numEpoch', type=int, default=1000, help='网络训练轮数')
parser.add_argument('--minEpoch', type=int, default=100, help='网络至少训练轮数')
parser.add_argument('--random_seed', type=int, default=0, help='固定随机数种子')
parser.add_argument('--sys_msg', type=str, default='ASPC', help='系统信息用于绘图')
args = parser.parse_args()
torch.backends.cudnn.benchmark = True
sys_random_fixed(args.random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print('Your device is', device)


def correcte_gt_by_idx(gt_1d, idx_1d, seg_gt_1d):
    """
    根据训练集和分割结果扩充获得伪训练集
    输入：所有标签，训练集坐标，分割结果
    输出：新训练集坐标、新标签、剩下的背景区域
    """
    seg_gt_1d += 100
    seg_lab = [np.where(seg_gt_1d == u_label) for u_label in np.unique(seg_gt_1d)]  # 存储每一类的各坐标值(一维)
    print('Pieces:', len(seg_lab))
    assert len(seg_lab) < 10000, "Pieces so many !"  # 如果分块数过多则提醒调整参数
    for idx in idx_1d:  # 遍历目标像素,将该区域赋值
        seg_gt_1d[idx] = gt_1d[idx]
    gt = np.zeros(seg_gt_1d.shape)
    to_delete = []
    for idx, inds in enumerate(seg_lab):  # 遍历这固定的每个小区域
        u_labels, hist = np.unique(seg_gt_1d[inds], return_counts=True)  # 对该小区域的每个标签进行计数
        if len(u_labels) >= 2:  # 如果该区域有训练集
            gt[inds] = (u_labels[:-1])[np.argmax(hist[:-1])]  # 由这个小区域的非零众数作为整个区域的值
            to_delete.append(idx)  # 有训练集的区域剔除掉
    seg_back = np.delete(np.asarray(seg_lab), to_delete)
    idx = np.where(gt > 0)
    return np.asarray(idx).squeeze(), np.asarray(gt).squeeze(), seg_back


def hsi_complex(data):
    """
    计算高光谱数据的复杂度：边缘数量除以空间面积 [(edge_sum/area)/50%]*200
    """
    data = (dataNormalize(data) * 255).astype('uint8')
    img = cv2.GaussianBlur(data, (5, 5), 10)
    edge = cv2.Canny(img, 10, 50)
    edge_sum = edge.sum() // 255
    data_complex = edge_sum.astype('float') / (edge.shape[0] * edge.shape[1])  # 边缘数量除以面积
    assert data_complex > 0, "Check data_complex !"
    print('hsi complex coefficient is %f' % data_complex)
    seg_coef = (1 - (data_complex / 0.50)) * 200  # 换算成分割算法的系数
    return seg_coef.astype('int')


def felzenszwalb_hsi_by_cut_4_pieces(hsi):
    """ 对原始数据先分块再分割最后合并 level 2 in paper"""

    def dataset_cut_4pieces(data):
        """ 对数据集四分块，同时控制长宽分别为16的倍数 """
        height = (data.shape[0] + 4 - 1) // 4 * 2  # 获取块的长宽(分别控制为16的倍数)
        width = (data.shape[1] + 4 - 1) // 4 * 2
        data = data[np.newaxis]
        data_pieces = data[:, :height, :width]
        data_pieces = np.vstack((data_pieces, data[:, :height, -width:]))
        data_pieces = np.vstack((data_pieces, data[:, -height:, :width]))
        data_pieces = np.vstack((data_pieces, data[:, -height:, -width:]))
        return data_pieces.astype('float32')

    def dataset_merge_4pieces(pieces, row, col):
        """ 将四分块数据合成为一张大图 """
        center_h, center_w = row // 2, col // 2
        center_h_, center_w_ = (row + 1) // 2, (col + 1) // 2
        data = np.zeros((row, col))
        data[:center_h, :center_w] = pieces[0][:center_h, :center_w]
        data[:center_h, -center_w_:] = pieces[1][:center_h, -center_w_:]
        data[-center_h_:, :center_w] = pieces[2][-center_h_:, :center_w]
        data[-center_h_:, -center_w_:] = pieces[3][-center_h_:, -center_w_:]
        return np.asarray(data).astype('long')

    R, C, _ = hsi.shape
    seg_gt_pieces = []
    data_pieces = dataset_cut_4pieces(hsi)  # 对原始数据分块
    for i, piece in enumerate(data_pieces):  # 依次处理各小块
        seg_piece = felzenszwalb(data_hsi, min_size=hsi_complex(piece)) + 10000 * i  # 区别每个块
        seg_gt_pieces.append(seg_piece)  # 对该块单独分割
    seg_gt = dataset_merge_4pieces(seg_gt_pieces, R, C)  # 将每个小块的结果合成大图
    return np.array(seg_gt)


""" 获取原始数据及基本信息 """
data_hsi = matDataLoad('./datasets/' + args.dataset + '.mat')
gt_hsi = matDataLoad('./datasets/' + args.dataset + '_gt.mat')
gt_hsi_1D = gt_hsi.flatten()
train_idx, test_idx = split_dataset_equal(gt_hsi_1D, args.numTrain)  # 分割训练集和测试集,每类数量一致

Row, Col, Layers = data_hsi.shape
NumClass = gt_hsi.max()
Model_in = Layers  # 模型输入
Overfit = (NumClass * args.numTrain - 1) / (NumClass * args.numTrain)

""" 分块图像分割获得最佳伪训练集 """
seg_gt_hsi = felzenszwalb_hsi_by_cut_4_pieces(data_hsi)  # 获得分割结果，包含分块处理
train_idx_spa, gt_hsi_1D_spa, seg_back = correcte_gt_by_idx(gt_hsi_1D, train_idx, seg_gt_hsi.flatten())  # 获得伪训练集

""" 数据转换 """
train_target = gt_hsi_1D[train_idx] - 1  # 真训练集标签
test_target = gt_hsi_1D[test_idx] - 1  # 真验证集标签
train_target_spa = gt_hsi_1D_spa[train_idx_spa] - 1  # 伪训练集标签
seg_back_idx = np.asarray([item for sublist in seg_back for item in sublist])  # 获得无标签背景区域的坐标
data_hsi_tensor = data_hsi.transpose(2, 0, 1)  # 转为band,row,col
data_hsi_tensor = data_hsi_tensor[np.newaxis, :, :, :]  # 1*103*610*340
data_hsi_tensor = torch.from_numpy(data_hsi_tensor).to(device)  # 转化为tensor数据
train_target_tensor = torch.from_numpy(train_target).long().to(device)
training_target_spa_tensor = torch.from_numpy(train_target_spa).long().to(device)

""" 建立模型 """
model = HRNet(Model_in, NumClass).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dec_epoch, gamma=args.lr_dec_rate)  # 动态学习率
model.train()
flag = 0  # 结束训练标志位
for batch_idx in range(args.numEpoch):
    ''' 计算训练集与伪训练集损失 '''
    optimizer.zero_grad()
    output = model(data_hsi_tensor)  # 输入1*103*610*340.输出 1*9*610*340
    output = output[0].permute(1, 2, 0)  # 转为610*340*9
    output = output.view(-1, NumClass)  # 展平为207400*9
    output_ = output
    output = torch.nn.functional.softmax(output, dim=-1)  # 归一化
    loss = criterion(output[train_idx], train_target_tensor)  # 计算真实训练集的损失
    loss_spa = criterion(output[train_idx_spa], training_target_spa_tensor)  # 计算伪训练集的损失
    """ 计算无标签伪训练集损失 """
    target_spa_back = torch.argmax(output, -1).data.cpu().numpy()  # 获得最大值索引，207400*1
    for i, inds in enumerate(seg_back):  # 遍历所有无标签背景区域
        u_labels, hist = np.unique(target_spa_back[inds], return_counts=True)  # 对该小区域的每个标签计数
        target_spa_back[inds] = u_labels[np.argmax(hist)]  # 由这个小区域的众数作为整个区域的值(临时标签)
    target_spa_back_ = torch.from_numpy(target_spa_back).long().to(device)
    loss_spa_back = criterion(output[seg_back_idx], target_spa_back_[seg_back_idx])  # 计算无标签伪训练集的损失
    loss = loss * 1.0 + loss_spa * 0.8 + loss_spa_back * 0.05  # 以不同权重结合三个损失
    loss.backward()
    optimizer.step()
    scheduler.step()  # 实时更新学习率
    ''' 打印本次训练的相关信息 '''
    im_target = torch.argmax(output, -1).data.cpu().numpy()  # 获得最大值索引numpy形式，207400*1
    acc_train = np.sum(im_target[train_idx] == train_target) / len(train_target)
    acc_test = np.sum(im_target[test_idx] == test_target) / len(test_target)
    print('Epoch:', batch_idx + 1, '/', args.numEpoch,
          'Loss:%.3f' % loss.item(), 'lr:%.3f' % (scheduler.get_lr()[0]),
          'Acc on train:%.3f' % acc_train, ',test:%.5f' % acc_test)
    if acc_train >= 1:  # 过拟合停止训练
        flag += 1
        if flag > 20:
            if batch_idx >= args.minEpoch:
                break

import scipy.io as sio
from sklearn.metrics import confusion_matrix
import pandas as pd
from model import DL
import get_cls_map
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from model import LPDecoder_ogb as LPDecoder
from model import GCN_mgaev3 as GCN
import argparse
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import math
import random
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import (negative_sampling, add_self_loops)
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
import time
import numpy as np

def create_predictor(args, device):

    predictor = LPDecoder(args.hidden_channels, args.decode_channels, 1, args.num_layers,
                              args.decode_layers, args.s2gae_dropout).to(device)
    return predictor

def ss_loadData(feature, labels, x0, x1, args):
    x0 = int(x0)
    x1 = int(x1)
    data =feature.reshape( x0, x1, args.num_layers*128)
    labels = labels.reshape(x0, x1)
    return data, labels

def supervised_test(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))
    return y_pred_test, y_test

def supervised_train(train_loader, num, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = DL().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    total_loss = 0
    for epoch in range(args.classifier_epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch == 0:
            print("\n\033[92m有监督的分类器训练是：绿色\033[0m")
        print('\033[92m[Runs: %d/%d]   [Epoch: %d/%d]   [loss avg: %.4f]   [current loss: %.4f]\033[0m' % (num+1, args.num_run ,epoch + 1, args.classifier_epochs ,
                                                                                        total_loss / (epoch + 1),
                                                                                        loss.item()))
    print('Finished Training')
    return net, device

def ss_create_data_loader(feature, labels, x0, x1, args):
    X, y = ss_loadData(feature, labels ,x0 ,x1,args)
    y_out = y
    index = np.nonzero(y.reshape(y.shape[0] * y.shape[1]))[0]

    test_ratio = 0.9


    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=args.pca_components)
    print('Data shape after PCA: ', X_pca.shape)
    print('\n... ... create data cubes ... ...')

    X_pca, y, X_pca_whole, y_whole = createImageCubes(X_pca, y, windowSize=args.patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)
    print('Pixel after removing the unlabelled areas :', X_pca.shape[0])

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
    Xtest = X_pca
    ytest = y
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    X = X_pca.reshape(-1, args.patch_size, args.patch_size, args.pca_components, 1)
    Xtrain = Xtrain.reshape(-1, args.patch_size, args.patch_size, args.pca_components, 1)
    Xtest = Xtest.reshape(-1, args.patch_size, args.patch_size, args.pca_components, 1)
    X_whole = X_pca_whole.reshape(-1, args.patch_size, args.patch_size, args.pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    X_whole = X_whole.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    X = TestDS(X, y)
    X_whole = TestDS(X_whole, y_whole)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)

    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True
                                               )

    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              drop_last=False
                                              )

    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  drop_last=False
                                                  )

    all_data_loader_whole = torch.utils.data.DataLoader(dataset=X_whole,
                                                        batch_size=args.batch_size,
                                                        shuffle=False,
                                                        num_workers=0,
                                                        drop_last=False
                                                        )

    return train_loader, test_loader, y, index, all_data_loader, all_data_loader_whole, y_out

def construct_edges_and_features(HSI_data):
    features = []
    edges = []

    for i in range(HSI_data.shape[0]):
        for j in range(HSI_data.shape[1]):
            node_index = i * HSI_data.shape[1] + j
            features.append(HSI_data[i, j, :])
            if i > 0:
                edges.append((node_index, node_index - HSI_data.shape[1]))
            if i < HSI_data.shape[0] - 1:
                edges.append((node_index, node_index + HSI_data.shape[1]))
            if j > 0:
                edges.append((node_index, node_index - 1))
            if j < HSI_data.shape[1] - 1:
                edges.append((node_index, node_index + 1))

    return features, edges

def extract_feature_list_layer2(feature_list):
    xx_list = []
    xx_list.append(feature_list[-1])
    tmp_feat = torch.cat(feature_list, dim=-1)
    xx_list.append(tmp_feat)
    return xx_list

def edgemask_random(mask_ratio, split_edge, device, num_nodes, height, width,):
    if isinstance(split_edge, torch.Tensor):
        edge_index = to_undirected(split_edge.t()).t()

    else:
        edge_index = torch.stack([split_edge['train']['edge'][:, 1], split_edge['train']['edge'][:, 0]], dim=1)
        edge_index = torch.cat([split_edge['train']['edge'], edge_index], dim=0)

    num_edge = len(edge_index)
    index = np.arange(num_edge)#边的乱序索引
    np.random.shuffle(index)
    mask_num = int(num_edge * mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num])
    mask_index = torch.from_numpy(index[-mask_num:])
    pre_index = pre_index.long()
    edge_index_train = edge_index[pre_index].t()
    mask_index = mask_index.long()
    edge_index_mask = edge_index[mask_index].to(device)

    edge_index = edge_index_train
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index).t()
    return adj, edge_index, edge_index_mask.to(device)

def self_supervised_train(model, predictor, x, edge_index, optimizer, height, width, args):

    model.train()
    predictor.train()

    total_loss = total_examples = 0

    adj, _, pos_train_edge = edgemask_random(args.mask_ratio, edge_index, x.device, x.shape[0], height, width)


    adj = adj.to(x.device)

    for perm in DataLoader(range(pos_train_edge.size(0)), args.s2gae_batch_size, shuffle=True):


        optimizer.zero_grad()

        h = model(x, adj)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h, edge)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        edge = torch.randint(0, x.shape[0], edge.size(), dtype=torch.long,
                             device=x.device)
        neg_out = predictor(h, edge)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test(model, predictor, x, pos_test_edge, neg_test_edge, batch_size, full_adj_t, epoch):
    model.eval()
    predictor.eval()

    h = model(x, full_adj_t)

    pos_test_edge = pos_test_edge.to(x.device)
    neg_test_edge = neg_test_edge.to(x.device)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h, edge).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h, edge).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    test_true = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)
    test_auc = roc_auc_score(test_true, test_pred)

    return test_auc

def do_edge_split_nc(edge_index, num_nodes, val_ratio=0.05, test_ratio=0.1):
    random.seed(234)
    torch.manual_seed(234)
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    test_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    train_pos_edge_index = torch.stack([r, c], dim=0)

    neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes,num_neg_samples=row.size(0))
    test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
    train_pos_edge = torch.cat([train_pos_edge_index, val_pos_edge_index], dim=1)
    return train_pos_edge.t(), test_pos_edge_index.t(), test_neg_edge_index.t()

def loadData():

    data = sio.loadmat('data/houstonU2018.mat')['houstonU']
    labels = sio.loadmat('data/houstonU2018.mat')['houstonU_gt']
    return data, labels

def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize, removeZeroLabels = True):

    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0

    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData_removed = patchesData[patchesLabels>0,:,:,:]
        patchesLabels_removed = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return  patchesData_removed, patchesLabels_removed, patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)

    unique_classes = np.unique(y).astype(int)
    train_class_counts = {cls: np.sum(y_train == cls) for cls in unique_classes}
    test_class_counts = {cls: np.sum(y_test == cls) for cls in unique_classes}

    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    summary_df = pd.DataFrame({
        'Class No.': unique_classes,
        'Train': [train_class_counts[cls] for cls in unique_classes],
        'Test': [test_class_counts.get(cls, 0) for cls in unique_classes]
    })
    print(summary_df.to_string(index=False))
    return X_train, X_test, y_train, y_test

def create_data_loader():

    X, y= loadData()

    y_out = y
    index = np.nonzero(y.reshape(y.shape[0]*y.shape[1]))[0]

    test_ratio = 0.95

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=args.pca_components)
    print('Data shape after PCA: ', X_pca.shape)
    print('\n... ... create data cubes ... ...')

    X_pca, y, X_pca_whole, y_whole = createImageCubes(X_pca, y, windowSize=args.patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)
    print('Pixel after removing the unlabelled areas :', X_pca.shape[0])

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
    Xtest = X_pca
    ytest = y
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    X = X_pca.reshape(-1, args.patch_size, args.patch_size, args.pca_components, 1)
    Xtrain = Xtrain.reshape(-1, args.patch_size, args.patch_size, args.pca_components, 1)
    Xtest = Xtest.reshape(-1, args.patch_size, args.patch_size, args.pca_components, 1)
    X_whole  = X_pca_whole.reshape(-1, args.patch_size, args.patch_size, args.pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    X_whole = X_whole.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    X = TestDS(X, y)
    X_whole = TestDS(X_whole, y_whole)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)

    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True
                                               )

    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              drop_last=False
                                              )

    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  drop_last=False
                                                  )

    all_data_loader_whole = torch.utils.data.DataLoader(dataset=X_whole,
                                                        batch_size=args.batch_size,
                                                        shuffle=False,
                                                        num_workers=0,
                                                        drop_last=False
                                                        )

    return train_loader, test_loader, y, index, all_data_loader, all_data_loader_whole, y_out

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len

class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len

def train(train_loader, args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = DL().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    total_loss = 0
    for epoch in range(args.classifier_epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Runs: %d/%d]   [Epoch: %d/%d]   [loss avg: %.4f]   [current loss: %.4f]' % (num+1, args.num_run ,epoch + 1, args.classifier_epochs ,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))

    print('Finished Training')
    return net, device

def mytest(device, net, test_loader):

    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test
def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)

    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--S2GAE', default='y', help='Use S2GAE?--[y/n]')

    parser.add_argument('--pca_components', default=30)
    parser.add_argument('--patch_size', default=5, help='Typically take 15.')
    parser.add_argument('--batch_size', default=64, help='Typically take 64.')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--num_run', default=1, help='Typically take 10.')
    parser.add_argument('--classifier_epochs', default=100, help='Typically take 100.')

    parser.add_argument('--num_layers', type=int, default=3, help='Typically take 3.')
    parser.add_argument('--decode_layers', type=int, default=3, help='Typically take 3.')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Typically take 128.')
    parser.add_argument('--decode_channels', type=int, default=256, help='Typically take 256.')
    parser.add_argument('--s2gae_dropout', type=float, default=0.5, help='Typically take 0.5.')
    parser.add_argument('--s2gae_batch_size', type=int, default=1024*64)
    parser.add_argument('--s2gae_lr', type=float, default=0.0001)
    parser.add_argument('--s2gae_epochs', type=int, default=10)
    parser.add_argument('--mask_ratio', type=float, default=0.6)

    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.S2GAE == 'y':
        HSI_data, HSI_labels = loadData()

        n_rows, n_cols, n_bands = HSI_data.shape
        HSI_data = HSI_data.reshape((n_rows * n_cols, n_bands))
        pca = PCA(n_components=30)
        HSI_data = pca.fit_transform(HSI_data)
        HSI_data = HSI_data.reshape((n_rows, n_cols, 30))
        print(f"\n降维后的数据形状: {HSI_data.shape}")
        scaler = MinMaxScaler()
        HSI_data = HSI_data.reshape((n_rows * n_cols, 30))
        HSI_data = scaler.fit_transform(HSI_data)
        HSI_data = HSI_data.reshape((n_rows, n_cols, 30))
        print("归一化后的数据形状:", HSI_data.shape)

        features, edges = construct_edges_and_features(HSI_data)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        features = np.array(features)
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(HSI_labels, dtype=torch.int).reshape(-1)
        edge_index = to_undirected(edge_index)
        full_adj_t = SparseTensor.from_edge_index(edge_index).t()
        edge_index, test_edge, test_edge_neg = do_edge_split_nc(edge_index, x.shape[0])
        labels = y.view(-1)
        x = x.to(device)
        y = y.to(device)
        edge_index = edge_index.to(device)
        full_adj_t = full_adj_t.to(device)

        model = GCN(x.shape[1], args.hidden_channels, args.hidden_channels, args.num_layers,
                    args.s2gae_dropout).to(device)

        predictor = create_predictor(args, device)

        print('\nStart training with mask ratio={} # optimization edges={} / {}'.format(args.mask_ratio,
                                                                                        int(args.mask_ratio *
                                                                                            edge_index.shape[
                                                                                                0]),
                                                                                        edge_index.shape[0]))
        t1 = time.time()
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.s2gae_lr)
        best_valid = 0.0
        best_epoch = 0

        for epoch in range(1, 1 + args.s2gae_epochs):

            loss = self_supervised_train(model, predictor, x, edge_index, optimizer, n_rows, n_cols, args)

            auc_test = test(model, predictor, x, test_edge, test_edge_neg, args.s2gae_batch_size, full_adj_t, epoch)
            if auc_test > best_valid:
                best_valid = auc_test
                best_epoch = epoch

            if epoch == 1:
                print("\n\033[94m无监督的特征提取训练是：蓝色\033[0m")
            print(f'\033[94mRun: {1:02d}, 'f'Epoch: {epoch:02d}, 'f'Best_epoch: {best_epoch:02d}, 'f'Best_valid: {100 * best_valid:.2f}%, 'f'Loss: {loss:.4f}, \033[0m')

        t2 = time.time()
        self_time = t2 - t1
        print(f"extracting time: {self_time:.2f} seconds")
        feature = model(x, full_adj_t)
        feature = [feature_.detach() for feature_ in feature]
        feature_list = extract_feature_list_layer2(feature)

        for i, feature_tmp in enumerate(feature_list):
            if i == 1:
                feature, labels, x0, x1 = feature_tmp.data.cpu().numpy(), labels.data.cpu().numpy(), \
                                          HSI_data.shape[
                                              0], HSI_data.shape[1]
                oa = []
                acc = []
                aa = []
                kappa = []
                for num in range(args.num_run):
                    train_loader, test_loader, y_all, index, all_data_loader, all_data_loader_whole, y = ss_create_data_loader(
                            feature, labels, x0, x1, args)

                    tic1 = time.perf_counter()
                    net, device = supervised_train(train_loader, num, args)
                    toc1 = time.perf_counter()
                    tic2 = time.perf_counter()
                    y_pred_test, y_test = supervised_test(device, net, test_loader)
                    toc2 = time.perf_counter()
                    each_oa, confusion, each_acc, each_aa, each_kappa = acc_reports(y_test, y_pred_test)
                    oa.append(each_oa)
                    acc.append(each_acc)
                    aa.append(each_aa)
                    kappa.append(each_kappa)
                    print('\n Training time: %.2f seconds' % (toc1 - tic1))
                    print('\n Testing time: %.2f seconds' % (toc2 - tic2))

                print('\n', np.mean(acc, axis=0), np.std(acc, axis=0))
                print('\n OA: %.2f, std: %.2f ' % (np.mean(oa), np.std(oa)))
                print('\n AA: %.2f, std: %.2f ' % (np.mean(aa), np.std(aa)))
                print('\n Kappa: %.2f, std: %.2f ' % (np.mean(kappa), np.std(kappa)))

                get_cls_map.get_cls_map(net, device, all_data_loader, all_data_loader_whole, y)
    else:
        oa = []
        acc = []
        aa = []
        kappa = []
        for num in range(args.num_run):

            train_loader, test_loader, y_all, index, all_data_loader, all_data_loader_whole, y = create_data_loader()

            tic1 = time.perf_counter()
            net, device = train(train_loader, args)
            toc1 = time.perf_counter()
            tic2 = time.perf_counter()
            y_pred_test, y_test = mytest(device, net, test_loader)
            toc2 = time.perf_counter()
            each_oa, confusion, each_acc, each_aa, each_kappa = acc_reports(y_test, y_pred_test)
            oa.append(each_oa)
            acc.append(each_acc)
            aa.append(each_aa)
            kappa.append(each_kappa)
            print('\n Training time: %.2f seconds' % (toc1 - tic1))
            print('\n Testing time: %.2f seconds' % (toc2 - tic2))

        print('\n', np.mean(acc, axis=0), np.std(acc, axis=0))
        print('\n OA: %.2f, std: %.2f ' % (np.mean(oa), np.std(oa)))
        print('\n AA: %.2f, std: %.2f ' % (np.mean(aa), np.std(aa)))
        print('\n Kappa: %.2f, std: %.2f ' % (np.mean(kappa), np.std(kappa)))

        get_cls_map.get_cls_map(net, device, all_data_loader, all_data_loader_whole, y)
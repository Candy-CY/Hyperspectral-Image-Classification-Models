import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import torch.nn as nn
import torch.functional as F
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
## GLOBAL VARIABLES
dataset = 'IP'
test_ratio = 0.99
train_val_ratio = 1
train_ratio = 1-test_ratio
windowSize = 11
if dataset == 'UP':
    componentsNum = 15
elif dataset == 'UH':
    componentsNum = 50 if test_ratio >= 0.99 else 25
elif dataset == 'IP':
    componentsNum = 110
else:
    componentsNum = 30
drop = 0.4
## define a series of data progress function
def loadData(name):
    data_path = os.path.join(os.getcwd(),'../input/hybridsn/data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'UP':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'UH':
        data = sio.loadmat(os.path.join(data_path, 'HoustonU.mat'))['houstonU'] # 601*2384*50
        labels = sio.loadmat(os.path.join(data_path, 'HoustonU_gt.mat'))['houstonU_gt']
    return data, labels
def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,stratify=y)
    return X_train, X_test, y_train, y_test
def applyPCA(X, numComponents=64):
    newX = np.reshape(X, (-1, X.shape[2]))
    print(newX.shape)
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca, pca.explained_variance_ratio_
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]),dtype="float16")
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
def createPatches(X, y, windowSize=25, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]),dtype="float16")
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]),dtype="float16")
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels
def infoChange(X,numComponents):
    X_copy = np.zeros((X.shape[0] , X.shape[1], X.shape[2]))
    half = int(numComponents/2)
    for i in range(0,half-1):
        X_copy[:,:,i] = X[:,:,(half-i)*2-1]
    for i in range(half,numComponents):
        X_copy[:,:,i] = X[:,:,(i-half)*2]
    X = X_copy
    return X
# implementation of covariance pooling layers
def cov_pooling(features):
    shape_f = features.shape.as_list()
    features = features.transpose(1,2)
    centers_batch = features.mean(axis=2,keepdim=True)
    centers_batch = torch.reshape(centers_batch,(-1, 1, shape_f[2]))
    centers_batch = centers_batch.repeat(-1, 1, shape_f[2])

    return
# Data processing
X, y = loadData(dataset)
#print('X shape: ', X.shape)
#print('y shape: ', y.shape)
X,pca,ratio = applyPCA(X,numComponents=componentsNum)
#print('X after PCA shape: ', X.shape)
X = infoChange(X,componentsNum) # channel-wise shift
#print('X after infoChange shape: ', X.shape)
X, y = createPatches(X, y, windowSize=windowSize)
#print('Data cube X shape: ', X.shape)
#print('Data cube y shape: ', y.shape)
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)
#print('Xtrain shape: ', Xtrain.shape)
#print('Xtest  shape: ', Xtest.shape)

## Train
if dataset == 'UP':
    output_units = 9
elif dataset == 'UH':
    output_units = 20
else:
    output_units = 16
Xtrain = Xtrain.reshape(-1, windowSize, windowSize, componentsNum, 1)
#print('Xtrain shape: ', Xtrain.shape)
Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
#print('After transpose Xtrain shape: ', Xtrain.shape)
#ytrain = np_utils.to_categorical(ytrain)#将类别向量转换为二进制(只有0和1)的矩阵类型表示
ytrain = torch.nn.functional.one_hot(torch.from_numpy(ytrain.astype(np.int64)),
                                     num_classes=output_units).numpy()
Xvalid, Xtest, yvalid, ytest = splitTrainTestSet(Xtest, ytest, (test_ratio-train_ratio/train_val_ratio)/test_ratio)
Xvalid = Xvalid.reshape(-1, windowSize, windowSize, componentsNum, 1)
#print('Xvalid shape: ', Xtest.shape)
Xvalid  = Xvalid.transpose(0, 4, 3, 1, 2)
#print('After transpose Xvalid shape: ', Xtrain.shape)
#yvalid = np_utils.to_categorical(yvalid)
yvalid = torch.nn.functional.one_hot(torch.from_numpy(yvalid.astype(np.int64)),
                                     num_classes=output_units).numpy()
## input layer
#input_layer = Input((windowSize, windowSize, componentsNum, 1))
""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xvalid.shape[0]#Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xvalid)#Xtest)
        self.y_data = torch.LongTensor(yvalid)#ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len

# 创建 trainloader 和 testloader
trainset = TrainDS()
testset  = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset,batch_size=128)#, shuffle=True, num_workers=2)
test_loader  = torch.utils.data.DataLoader(dataset=testset,batch_size=128)#, shuffle=False, num_workers=2)
#print('train_loader shape: ', train_loader)
#print('test_loader shape: ', test_loader)
## convolutional layers
class MCNN(nn.Module):
    def __init__(self):
        super(MCNN,self).__init__()# 继承 __init__()功能
        #定义第一层 3D卷积层
        self.conv3d_1 = nn.Sequential(
            #输入参数的形式是：(windowSize, windowSize, componentsNum, 1) ========> (11, 11, 110, 8)
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True),
        )
        self.conv3d_2 = nn.Sequential(
            # 输入的结果是(11,11,110,8) ========> (11, 11, 110, 16)
            nn.Conv3d(8,16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
        )
        self.conv3d_3 = nn.Sequential(
            # 输入的结果是(11,11,110,16) ========> (11, 11, 110, 32)
            nn.Conv3d(16,32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True),
        )
        self.conv2d_4 = nn.Sequential(
        nn.Conv2d(3520, 64, kernel_size=(3, 3), stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True),
        )
        self.fc1 = nn.Linear(7744,256)#4096,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,16)
        self.dropout = nn.Dropout(p = 0.4)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.conv3d_1(x)
        out = self.conv3d_2(out)
        out = self.conv3d_3(out)
        out = self.conv2d_4(out.reshape(out.shape[0],-1,11,11))
        #out = cov_pooling(out)
        #out = feature_vector(out)
        out = out.reshape(out.shape[0],-1)
        out = self.relu(self.dropout(self.fc1(out)))
        out = self.relu(self.dropout(self.fc2(out)))
        out = self.fc3(out)
        return out

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 网络放到GPU上
net = MCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)#设置学习率
# 计时
import time
time_start=time.time()
# 开始训练
total_loss = 0
for epoch in range(2):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 优化器梯度归零
        optimizer.zero_grad()
        # 正向传播 +　反向传播 + 优化
        outputs = net(inputs)
        print("*1"*20)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))

time_end=time.time()
print('totally cost',time_end-time_start)
print('Finished Training')

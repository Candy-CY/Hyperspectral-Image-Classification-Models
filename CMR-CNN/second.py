import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
# import cv2 as cv
import matplotlib.pyplot as plt
import spectral
from torch.nn import init
# from CenterLoss import CenterLoss
# import seaborn as sns
import math
#读入数据
# !wget http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat
# !wget http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat
# !pip install spectral

#引入库函数
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,cohen_kappa_score
import spectral
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from operator import truediv


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RestNetBasicBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        # output = self.cbam(output)

        output = self.conv2(output)
        output = self.bn2(output)
        # output = self.cbam(output)

        return F.relu(x + output)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RestNetBasicBlock_3D(nn.Module):                 #(8,30,28,,28)                 
    def __init__(self, in_channels, out_channels):
        super(RestNetBasicBlock_3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.extra = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        
        output = self.conv1(x)
        output = F.relu(self.bn1(output))

        output = self.conv2(output)
        output = self.bn2(output)

        return F.relu(x + output)


class PyramidFeatures(nn.Module):
    """Feature pyramid module with top-down feature pathway"""
    def __init__(self, B2_size, B3_size, B4_size, B5_size, feature_size=128):
        super(PyramidFeatures, self).__init__()

        self.P5_1 = nn.Conv2d(B5_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(feature_size, 128, kernel_size=3, stride=1, padding=1)
        self.P4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):  #(240,28,)
        B3, B4, B5 = inputs

        P5_x = self.P5_1(B5)
        P5_x = self.P5_2(P5_x)
        # P4_x = self.P4_1(P5_x)
        # P4_x = self.P4_2(P4_x)

        return  P5_x


class HybridSN(nn.Module):
    def __init__(self):
        super(HybridSN, self).__init__()

        self.fpn = PyramidFeatures(64, 240, 208, 192)
        layer = [3, 4, 6, 3]
        self.inplanes = 128

        #三维卷积
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(1, 1, 1), stride=(2, 1, 1), padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace = True)
            )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=0),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True)
            ) 
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True)
        )

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True)
            )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True)
            ) 

        self.layer2 = self._make_layer(BasicBlock, 128, layer[0], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 512, layer[2], stride=1)
        # self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layer1 = RestNetBasicBlock(96, 128)
        self.layer1_3d = RestNetBasicBlock_3D(in_channels=32, out_channels=32)
        self.layer2_3d = RestNetBasicBlock_3D(in_channels=16, out_channels=16)
        self.layer3_3d = RestNetBasicBlock_3D(in_channels=8, out_channels=8)

        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(128, 16)
        self.dropout = nn.Dropout(p = 0.4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, plane, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != plane * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, plane * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(plane * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, plane, stride, downsample))
        self.inplanes = plane * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, plane))

        return nn.Sequential(*layers)

    def forward(self,x):

        # print('x = ', x.size())
        out_1 = self.conv3d_1(x)
        out_1 = self.layer3_3d(out_1)
        # out_1 = self.cbam_1(out_1)

        out_2 = self.conv3d_2(out_1)
        out_2 = self.layer2_3d(out_2)
        # out_2 = self.cbam_2(out_2)

        out_3 = self.conv3d_3(out_2)
        out_3 = self.layer1_3d(out_3)

        # print('----', out_3.size())

        #out_1 = out_3.reshape(out_3.shape[0], -1, 28, 28)
        out_1 = out_3.reshape(out_3.shape[0], -1, 28, 28)
        # out_2 = out_2.reshape(out_2.shape[0], -1, 14, 14)
        # out_3 = out_3.reshape(out_3.shape[0],-1,7,7)

        # out_2 = self.fpn([out_1, out_2, out_1])
        out_2 = self.layer1(out_1)
        # print('out_2=', out_2.size())
        out_3 = self.layer2(out_2)
        # out_3 = self.layer3(out_3)
        #print('out_3=', out_3.size())

        out_avg = self.avgpool(out_3)
        out = out_avg.view(out_avg.size(0), -1)

        # out1 = self.dropout(self.fc1(out))
        # out = self.fc2(out1)

        out = self.fc3(out)

        return out, out


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

#对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

#在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=3, removeZeroLabels = True):
    # 给 X 做 padding
    margin = int((windowSize ) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin, c - margin:c + margin]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    
    print('len_x_train = ', len(X_train))
    print('len_X_test = ', len(X_test))
    print('len_y_train = ', len(y_train))
    print('len_y_test = ', len(y_test))

    return X_train, X_test, y_train, y_test

# 地物类别
class_num = 16
#class_num = 9
X = sio.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']

#X = sio.loadmat('PaviaU.mat')['paviaU']
#y = sio.loadmat('PaviaU_gt.mat')['paviaU_gt']

# 用于测试样本的比例
test_ratio = 0.95
# 每个像素周围提取 patch 的尺寸
patch_size = 28
# 使用 PCA 降维，得到主成分的数量
pca_components = 30

print('Hyperspectral data shape: ', X.shape)
print('Label shape: ', y.shape)

print('\n... ... PCA tranformation ... ...')
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA: ', X_pca.shape)

print('\n... ... create data cubes ... ...')
X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
print('Data cube X shape: ', X_pca.shape)
print('Data cube y shape: ', y.shape)

print('\n... ... create train & test data ... ...')
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
print('Xtrain shape: ', Xtrain.shape)
print('Xtest  shape: ', Xtest.shape)

# 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
Xtest  = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
print('before transpose: Xtrain shape: ', Xtrain.shape) 
print('before transpose: Xtest  shape: ', Xtest.shape) 

# 为了适应 pytorch 结构，数据要做 transpose
Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
Xtest  = Xtest.transpose(0, 4, 3, 1, 2)
print('after transpose: Xtrain shape: ', Xtrain.shape) 
print('after transpose: Xtest  shape: ', Xtest.shape) 


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
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

# 创建 trainloader 和 testloader
trainset = TrainDS()
testset  = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=100, shuffle=True, num_workers=4)
test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=100, shuffle=False, num_workers=4)

# print('trainset', len(trainset))
# print('testset', len(testset))

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_dir = os.path.expanduser(os.getenv('TORCH_HOME', 'Final2_Model.pth'))
# filename = os.path.basename(urlparse(url).path)
pretrained_path = os.path.join(model_dir)

# 网络放到GPU上
net = HybridSN().to(device)

# if pretrained_path:
#     # print('--------------------------------------------------')
#     logging.info('load pretrained backbone')
#     net_dict = net.state_dict()
#     pretrained_dict = torch.load(pretrained_path)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
#     net_dict.update(pretrained_dict)
#     net.load_state_dict(net_dict)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=5e-4, weight_decay=2e-5)
max_testacc = 0

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# 开始训练
total_loss = 0
def test():
    net.eval()
    accs   = np.ones((len(test_loader))) * -1000.0
    losses = np.ones((len(test_loader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(test_loader):

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        _, outputs = net(inputs)
        losses[batch_idx] = criterion(outputs, targets).item()
        accs[batch_idx] = accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return (np.average(losses), np.average(accs))
acc = []
epoch_number = []
ip1_loader = []
idx_loader = []
#centerloss = CenterLoss(16, 16).to(device)

for epoch in range(100):

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 优化器梯度归零
        optimizer.zero_grad()
        # 正向传播 +　反向传播 + 优化 
        idx, outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        ip1_loader.append(outputs)
        idx_loader.append((labels))

    # feat = torch.cat(ip1_loader, 0)
    # labels = torch.cat(idx_loader, 0)
    # visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch)

    test_loss, test_acc = test()
    if test_acc > max_testacc:
        max_testacc = test_acc
        torch.save(net.state_dict(), 'Final1_Model.pth')
    epoch_number.append([epoch])
    acc.append([max_testacc])
    # print(epoch_number

    print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))


torch.save(net.state_dict(), 'Final2_Model.pth')
plt.plot(epoch_number,acc,color='r',label='acc') # r表示红色


#####非必须内容#########

plt.xlabel('epochs') #x轴表示

plt.ylabel('y label') #y轴表示

plt.title("chart") #图标标题表示

plt.legend() #每条折线的label显示

#######################

plt.savefig('test.jpg') #保存图片，路径名为test.jpg

print('Finished Training')
# net.load_state_dict(torch.load('Final1_Model.pth'))
count = 0
# 模型测试
for inputs, _ in test_loader:
    inputs = inputs.to(device)
    _, outputs = net(inputs)
    outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
    if count == 0:
        y_pred_test =  outputs
        count = 1
    else:
        y_pred_test = np.concatenate( (y_pred_test, outputs) )
# 生成分类报告
classification = classification_report(ytest, y_pred_test, digits=4)
print(classification)


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports (test_loader, y_test, name):
    count = 0
    # 模型测试
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        _, outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred =  outputs
            count = 1
        else:
            y_pred = np.concatenate( (y_pred, outputs) )

    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                        ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']
    
    classification = classification_report(y_test, y_pred, target_names=target_names)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    return classification, confusion, oa*100, each_acc*100, aa*100, kappa*100
#将结果写在文件里

classification, confusion, oa, each_acc, aa, kappa = reports(test_loader, ytest, 'IP')

# a = sns.heatmap(confusion, fmt='g', cmap='Blues', cbar=False)
# a = a.get_figure()
# a.savefig('123', dpi=400)
# plt.show()

classification = str(classification)
confusion = str(confusion)
file_name = "classification_report_IP_5.txt"

with open(file_name, 'w') as x_file:
    x_file.write('\n')
    x_file.write('{} Kappa accuracy (%)'.format(kappa))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%)'.format(oa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(aa))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion))

net.eval()
#显示结果
# load the original image
X = sio.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']

#X = sio.loadmat('PaviaU.mat')['paviaU']
#y = sio.loadmat('PaviaU_gt.mat')['paviaU_gt']

height = y.shape[0]
width = y.shape[1]

X = applyPCA(X, numComponents= pca_components)
X = padWithZeros(X, patch_size//2)

# 逐像素预测类别
outputs = np.zeros((height,width))
for i in range(height):
    for j in range(width):
        if int(y[i,j]) == 0:
            continue
        else :
            image_patch = X[i:i+patch_size, j:j+patch_size, :]
            image_patch = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2], 1)
            X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)                                   
            _, prediction = net(X_test_image)
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction+1
    if i % 20 == 0:
        print('... ... row ', i, ' handling ... ...')
predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(5,5), bands=[69, 27, 11])

spectral.save_rgb('predict_image_ip_5.jpg', outputs)

plt.pause(180)

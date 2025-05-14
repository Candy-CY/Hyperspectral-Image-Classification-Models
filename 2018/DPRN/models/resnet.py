import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    # 两层两个 3×3卷积模块（包括归一化层和激活函数层）
    # BasicBlock中没有通道的放大和缩小
    def __init__(self, inplanes, planes, stride=1, downsample=None, option=False):
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

class Bottleneck(nn.Module):
    # 三层卷积网络（1×1、3×3、1×1）分别用来压缩维度，卷积处理，恢复维度
    # inplane是输入的通道数，plane是输出的通道数，expansion是对输出通道数的倍乘
    # bottleneck就是不走寻常路，它的任务就是要对通道数进行压缩，再放大.
    # 于是，plane不再代表输出的通道数，而是block内部压缩后的通道数，输出通道数变为 plane*expansion。
    expansion = 16
    def __init__(self, inplanes, planes, stride=1, downsample=None, option=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    # resnet共有4个阶段，其中第一阶段为一个3x3的卷积处理，stride为1，然后经过池化处理，
    # 此时特征图的尺寸 与 输入的尺寸保持一致。接下来是3个阶段：也就是代码中的 layer1,layer2,layer3。
    # 这里用 _make_layer函数产生三个layer层，此函数中需要用户输入每个 layer的 block类型（基础版还是bottleneck版）
    def __init__(self, depth, num_classes, bottleneck=False):
        super(ResNet, self).__init__()        
        self.inplanes = 16
        n = (depth - 2) / 6
        if bottleneck == True:
            n = n * 2 / 3
            block = Bottleneck
        else:
            block = BasicBlock
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # downSample的作用于在残差连接时 将输入的图像的通道数变成和卷积操作的尺寸一致
        # 根据 ResNet的结构特点，一般只在每层开始时进行判断
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(  #通道数恢复成一致
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

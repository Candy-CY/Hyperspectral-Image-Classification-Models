import torch
import torch.nn as nn
import math
#from math import round
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)

class BasicBlock(nn.Module):
	outchannel_ratio = 1
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(inplanes)
		self.conv1 = conv3x3(inplanes, planes, stride)        
		self.bn2 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn3 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride
	def forward(self, x):
		out = self.bn1(x)
		out = self.conv1(out)        
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn3(out)
		if self.downsample is not None:
			shortcut = self.downsample(x)
			featuremap_size = shortcut.size()[2:4]
		else:
			shortcut = x
			featuremap_size = out.size()[2:4]
		batch_size = out.size()[0]
		residual_channel = out.size()[1]
		shortcut_channel = shortcut.size()[1]
		if residual_channel != shortcut_channel:
			padding = torch.autograd.Variable(torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).cuda())               
			out += torch.cat((shortcut, padding), 1)
		else:
			out += shortcut
		return out

class Bottleneck(nn.Module):
	outchannel_ratio = 4
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(inplanes)
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		if stride == 2:
			self.conv2 = nn.Conv2d(planes, planes, kernel_size=8, stride=stride, padding=3, bias=False)
		else:
			self.conv2 = nn.Conv2d(planes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
		self.bn3 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
		self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
	def forward(self, x):
		out = self.bn1(x)
		out = self.conv1(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn3(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.bn4(out)
		if self.downsample is not None:
			shortcut = self.downsample(x)
			featuremap_size = shortcut.size()[2:4]
		else:
			shortcut = x
			featuremap_size = out.size()[2:4]
		batch_size = out.size()[0]
		residual_channel = out.size()[1]
		shortcut_channel = shortcut.size()[1]
		if residual_channel != shortcut_channel:
			padding = torch.autograd.Variable(torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).cuda())               
			try:
				out += torch.cat((shortcut, padding), 1)
			except:
				print("ERROR",out.shape, shortcut.shape, padding.shape)
				exit()
		else:
			out += shortcut
		return out

class pResNet(nn.Module):
	def __init__(self, depth, alpha, num_classes, n_bands, avgpoosize, inplanes, bottleneck=False):
		super(pResNet, self).__init__()
		self.inplanes = inplanes
		if bottleneck == True:
			n = (depth - 2) // 9
			block = Bottleneck
		else:
			n = (depth - 2) // 6
			block = BasicBlock
		self.addrate = alpha / (3*n*1.0)
		self.input_featuremap_dim = self.inplanes
		self.conv1 = nn.Conv2d(n_bands, self.input_featuremap_dim,kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
		self.featuremap_dim = self.input_featuremap_dim 
		self.layer1 = self.pyramidal_make_layer(block, n)
		self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
		self.layer3 = self.pyramidal_make_layer(block, n, stride=2)
		self.final_featuremap_dim = self.input_featuremap_dim
		self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
		self.relu_final = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(avgpoosize)
		self.fc = nn.Linear(self.final_featuremap_dim, num_classes)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
	def pyramidal_make_layer(self, block, block_depth, stride=1):
		downsample = None
		if stride != 1:
			# or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
			downsample = nn.AvgPool2d((2,2), stride = (2, 2))
		layers = []
		self.featuremap_dim = self.featuremap_dim + self.addrate
		layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)),
							stride, downsample))
		for i in range(1, block_depth):
			temp_featuremap_dim = self.featuremap_dim + self.addrate
			layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio,
								int(round(temp_featuremap_dim)), 1))
			self.featuremap_dim  = temp_featuremap_dim
		self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		#x = F.dropout(x)
		x = self.bn1(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.bn_final(x)
		x = self.relu_final(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

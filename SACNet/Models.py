import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()        
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

def adjust_learning_rate(optimizer,base_lr, i_iter, max_iter, power=0.9):
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def scaled_l2(X, C, S):
    """
    scaled_l2 distance
    Args:
        X (b*n*d):  original feature input
        C (k*d):    code words, with k codes, each with d dimension
        S (k):      scale cofficient
    Return:
        D (b*n*k):  relative distance to each code
    Note:
        apparently the X^2 + C^2 - 2XC computation is 2x faster than
        elementwise sum, perhaps due to friendly cache in gpu
    """
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert S.numel() == C.shape[0], "scale, codeword num mismatch"

    b, n, d = X.shape
    X = X.view(-1, d)  # [bn, d]
    Ct = C.t()  # [d, k]
    X2 = X.pow(2.0).sum(-1, keepdim=True)  # [bn, 1]
    C2 = Ct.pow(2.0).sum(0, keepdim=True)  # [1, k]
    norm = X2 + C2 - 2.0 * X.mm(Ct)  # [bn, k]
    scaled_norm = S * norm
    D = scaled_norm.view(b, n, -1)  # [b, n, k]
    return D


def aggregate(A, X, C):
    """
    aggregate residuals from N samples
    Args:
        A (b*n*k):  weight of each feature contribute to code residual
        X (b*n*d):  original feature input
        C (k*d):    code words, with k codes, each with d dimension
    Return:
        E (b*k*d):  residuals to each code
    """
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert A.shape[:2] == X.shape[:2], "weight, input dim mismatch"
    X = X.unsqueeze(2)  # [b, n, d] -> [b, n, 1, d]
    C = C[None, None, ...]  # [k, d] -> [1, 1, k, d]
    A = A.unsqueeze(-1)  # [b, n, k] -> [b, n, k, 1]
    R = (X - C) * A  # [b, n, k, d]
    E = R.sum(dim=1)  # [b, k, d]
    return E

class DilatedFCN (nn.Module):   
    def __init__(self,num_features=103, num_classes=9, conv_features=64):
        super(DilatedFCN , self).__init__()      
        self.conv0 = nn.Conv2d(num_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=1,
                               bias=True)
        self.conv1 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=2,
                               bias=True)
        self.conv2 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=3,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)     
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)        
        self.conv_cls = nn.Conv2d(conv_features, num_classes, kernel_size=1, stride=1, padding=0,
                               bias=True)

    def forward(self, x):
        
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.avgpool(x)
        x = self.relu(self.conv2(x))        
        x = self.conv_cls(x)
        x = interpolation(x)         
        return x


class SACNet(nn.Module):   
    def __init__(self,num_features=103, num_classes=9, conv_features=64, trans_features=32,K=48,D=32):
        super(SACNet, self).__init__()

        self.conv0 = nn.Conv2d(num_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=1,
                               bias=True)
        self.conv1 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=2,
                               bias=True)
        self.conv2 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=3,#3
                               bias=True)

        self.alpha3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.beta3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.gamma3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.deta3 = nn.Conv2d(trans_features, conv_features, kernel_size=1, stride=1, padding=0,
                               bias=False)

        self.encoding = nn.Conv2d(conv_features, D, kernel_size=1, stride=1, padding=0,
                               bias=False)

        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.attention = nn.Linear(D,conv_features)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)    
    
        self.conv_cls = nn.Conv2d(conv_features*3, num_classes, kernel_size=1, stride=1, padding=0,
                               bias=True)

        self.drop = nn.Dropout(0.5)
        self.conv_features = conv_features
        self.trans_features = trans_features
        self.K = K
        self.D = D
        
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)
        self.BN = nn.BatchNorm1d(K)

    def forward(self, x):        
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        x = self.relu(self.conv0(x))
        conv1 = x

        x = self.relu(self.conv1(x))
        conv2 = x
        x = self.avgpool(x)

        x = self.relu(self.conv2(x))
        n,c,h,w = x.size()
        interpolation_context3 = nn.UpsamplingBilinear2d(size=x.shape[2:4])

        x_half = self.avgpool(x)
        n,c,h,w = x_half.size()
        alpha_x = self.alpha3(x_half)
        beta_x = self.beta3(x_half)
        gamma_x = self.relu(self.gamma3(x_half))

        alpha_x = alpha_x.squeeze().permute(1, 2, 0)
        #h*w x c
        alpha_x = alpha_x.view(-1,self.trans_features)
        #c x h*w
        beta_x = beta_x.view(self.trans_features,-1)
        gamma_x = gamma_x.view(self.trans_features,-1)

        context_x = torch.matmul(alpha_x,beta_x)
        context_x = F.softmax(context_x)

        context_x = torch.matmul(gamma_x,context_x)
        context_x = context_x.view(n,self.trans_features,h,w)
        context_x = interpolation_context3(context_x)

        deta_x = self.relu(self.deta3(context_x))
        x = deta_x + x

        Z = self.relu(self.encoding(x)).view(1,self.D,-1).permute(0, 2, 1) #n,h*w,D

        A = F.softmax(scaled_l2(Z,self.codewords,self.scale),dim=2) # b,n,k
        E = aggregate(A, Z, self.codewords) # b,k,d
        E_sum = torch.sum(self.relu(self.BN(E)),1) # b,d
        gamma = self.sigmoid(self.attention(E_sum)) # b,num_conv
        gamma = gamma.view(-1, self.conv_features, 1, 1)
        x = x + x * gamma
        context3 = interpolation(x)
        conv2 = interpolation(conv2)
        conv1 = interpolation(conv1)

        x = torch.cat((conv1,conv2,context3),1)
        x = self.conv_cls(x)
        
        return x


class SpeFCN(nn.Module):   
    def __init__(self,num_features=103, num_classes=9):
        super(SpeFCN, self).__init__()

        self.conv1 = nn.Conv2d(num_features, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.conv_cls = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0,
                               bias=True)

    def forward(self, x):        
        x = self.relu(self.conv1(x))
        conv1 = x
        x = self.relu(self.conv2(x))
        conv2 = x
        x = self.relu(self.conv3(x))
        conv3 = x

        x = self.conv_cls(conv1+conv2+conv3)        
        return x
       

class SpaFCN(nn.Module):   
    def __init__(self,num_features=103, num_classes=9):
        super(SpaFCN, self).__init__()

        self.conv1 = nn.Conv2d(num_features, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)   
        self.relu = nn.ReLU(inplace=True)
        self.conv_cls = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0,
                               bias=True)


    def forward(self, x):        
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        x = self.relu(self.conv1(x))
        conv1 = x
        x = self.avgpool(self.relu(self.conv2(x)))
        conv2 = x
        x = self.avgpool(self.relu(self.conv3(x)))
        conv3 = x
        
        x = self.conv_cls(conv1+interpolation(conv2)+interpolation(conv3))
        
        return x
       
class SSFCN(nn.Module):   
    def __init__(self,num_features=103, num_classes=9):
        super(SSFCN, self).__init__()
        self.spe_conv1 = nn.Conv2d(num_features, 64, kernel_size=1)
        self.spe_conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.spe_conv3 = nn.Conv2d(64, 64, kernel_size=1)

        self.spa_conv1 = nn.Conv2d(num_features, 64, kernel_size=1)
        self.spa_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.spa_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)    
        self.w_spe = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.w_spa = nn.Parameter(torch.Tensor(1), requires_grad=True)
        
        self.w_spe.data.uniform_(1, 2)
        self.w_spa.data.uniform_(1, 2)

        self.relu = nn.ReLU(inplace=True)
        self.conv_cls = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0,
                               bias=True)

    def forward(self, x):        
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])        
        hsi = x

        x = self.relu(self.spe_conv1(hsi))
        spe_conv1 = x
        x = self.relu(self.spe_conv2(x))
        spe_conv2 = x
        x = self.relu(self.spe_conv3(x))
        spe_conv3 = x
        spe = spe_conv1 + spe_conv2 + spe_conv3

        x = self.relu(self.spa_conv1(hsi))
        spa_conv1 = x
        x = self.avgpool(self.relu(self.spa_conv2(x)))
        spa_conv2 = x
        x = self.avgpool(self.relu(self.spa_conv3(x)))
        spa_conv3 = x
        spa = spa_conv1 + interpolation(spa_conv2) + interpolation(spa_conv3)
        
        x = self.conv_cls(self.w_spe*spe+self.w_spa*spa)        
        return x

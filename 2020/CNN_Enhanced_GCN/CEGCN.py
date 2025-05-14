import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,A:torch.Tensor):
        super(GCNLayer, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1= torch.nn.Parameter(torch.tensor([0.1],requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 =nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 =nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count=self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask=torch.ceil( self.A*0.00001)
        
        
    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat
    
    def forward(self, H, model='normal'):
        # # 方案一：minmax归一化
        # H = self.BN(H)
        # H_xx1= self.GCN_liner_theta_1(H)
        # A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        # if model != 'normal': A=torch.clamp(A,0.1) #This is a trick.
        # D_hat = self.A_to_D_inv(A)
        # A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        # output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        # output = self.Activition(output)
        
        # # 方案二：softmax归一化 (加速运算)
        H = self.BN(H)
        H_xx1= self.GCN_liner_theta_1(H)
        e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))# matmul()数组向量乘积
        zero_vec = -9e15 * torch.ones_like(e)
        A = torch.where(self.mask > 0, e, zero_vec)+ self.I
        if model != 'normal': A=torch.clamp(A,0.1) # This is a trick for the Indian Pines.
        # clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
        A = F.softmax(A, dim=1)
        # torch.mul(a, b) 是矩阵a和b对应位相乘，a和b的维度必须相等。
        output = self.Activition(torch.mm(A, self.GCN_liner_out_1(H)))
        
        return output,A

class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch,kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        # LeakyReLU是ReLU函数的一个变体，解决了ReLU函数存在的问题，α的默认往往是非常小的，比如0.01，这样就保证了Dead Neurons的问题。
        self.Act2 = nn.LeakyReLU()
        self.BN=nn.BatchNorm2d(in_ch)
        
    
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out

class CEGCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, model='normal'):
        super(CEGCN, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model=model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q
        
        layers_count=2
        
        # Spectra Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(128),)
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        
        # Pixel-level Convolutional Sub-Network
        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.CNN_Branch.add_module('CNN_Branch'+str(i),SSConv(128, 128,kernel_size=5))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))
        
        # Superpixel-level Graph Sub-Network
        self.GCN_Branch=nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.GCN_Branch.add_module('GCN_Branch'+str(i),GCNLayer(128, 128, self.A))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 64, self.A))

        # Softmax layer
        self.Softmax_linear =nn.Sequential(nn.Linear(128, self.class_count))
    
    def forward(self, x: torch.Tensor):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        (h, w, c) = x.shape
        
        # 先去除噪声
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))# torch.unsqueeze(a，N)：就是在a中指定位置N加上一个维数为1的维度。
        noise =torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x=noise  #直连
        
        clean_x_flatten=clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # 低频部分
        hx = clean_x
        
        # CNN与GCN分两条支路
        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))# spectral-spatial convolution
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])

        # GCN层 1 转化为超像素 x_flat 乘以 列归一化Q
        H = superpixels_flatten
        if self.model=='normal':
            for i in range(len(self.GCN_Branch)): H, _ = self.GCN_Branch[i](H)
        else:
            for i in range(len(self.GCN_Branch)): H, _ = self.GCN_Branch[i](H,model='smoothed')
            
            
        GCN_result = torch.matmul(self.Q, H)  # 这里self.norm_row_Q == self.Q
        
        # 两组特征融合(两种融合方式)
        Y = torch.cat([GCN_result,CNN_result],dim=-1)
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y


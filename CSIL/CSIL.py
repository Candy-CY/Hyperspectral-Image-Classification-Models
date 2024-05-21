from func import *
import os
from operator import truediv
import torch.utils.data as Data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as scio
from einops import rearrange, repeat
import torch
import torch.nn as nn
import cv2
import time
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import cohen_kappa_score
import xlwt
import xlrd
from xlutils.copy import copy
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchsummary import summary
import torch_optimizer as optim2
import time
import collections
from network import *
from sklearn.manifold import TSNE
import pandas as pd


MIN_NUM_PATCHES = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Residual_1(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class AttentionS(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class AttentionL(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, mask=None):
        b, n, _, h = *x.shape, self.heads

        # get k and v by chunking matrix
        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))  # x->q: [8, 50, 1024], context->k=v: [8, 9, 1024]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class TransformerL(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_1(PreNorm(dim, AttentionL(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual_1(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, context, mask=None):
        for attn, ff in self.layers:
            x = attn(x, context=context, mask=mask)  # AttentionL有context作为输入
            x = ff(x)
        return x

class TransformerS(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # depth = 6
            self.layers.append(nn.ModuleList([
                Residual_1(PreNorm(dim, AttentionS(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual_1(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class CSIL(nn.Module):
    def __init__(self, *, image_sizeL, patch_sizeL, image_sizeS, patch_sizeS, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_sizeL % patch_sizeL == 0, 'Image dimensions must be divisible by the patch size.'
        num_patchesL = (image_sizeL // patch_sizeL) ** 2
        patch_dimL = channels * patch_sizeL ** 2
        assert num_patchesL > MIN_NUM_PATCHES, f'your number of patches ({num_patchesL}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_sizeL = patch_sizeL

        self.pos_embeddingL = nn.Parameter(torch.randn(1, num_patchesL + 1, dim))
        self.patch_to_embeddingL = nn.Linear(patch_dimL, dim)
        self.cls_tokenL = nn.Parameter(torch.randn(1, 1, dim))
        self.dropoutL = nn.Dropout(emb_dropout)

        self.transformerL = TransformerL(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_cls_tokenL = nn.Identity()

        self.mlp_headL = nn.Sequential(
            nn.LayerNorm(dim),
        )

        assert image_sizeS % patch_sizeS == 0, 'Image dimensions must be divisible by the patch size.'
        num_patchesS = (image_sizeS // patch_sizeS) ** 2
        patch_dimS = channels * patch_sizeS ** 2
        assert num_patchesS > MIN_NUM_PATCHES, f'your number of patches ({num_patchesS}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_sizeS = patch_sizeS

        self.pos_embeddingS = nn.Parameter(torch.randn(1, num_patchesS + 1, dim))
        self.patch_to_embeddingS = nn.Linear(patch_dimS, dim)
        self.cls_tokenS = nn.Parameter(torch.randn(1, 1, dim))
        self.dropoutS = nn.Dropout(emb_dropout)

        self.transformerS = TransformerS(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_cls_tokenS = nn.Identity()

        self.mlp_headS = nn.Sequential(
            nn.LayerNorm(dim),
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.Multimlp_head = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, num_classes)
        )

        context_dim = channels * image_sizeS ** 2
        self.context_to_embedding = nn.Linear(context_dim, dim)

    # 用上面 __init__ 定义的各层将网络连接
    def forward(self, imgL, imgS, mask=None):
        RegionSize = patch_size_L
        RegionNum_col = int(imgL.shape[2] / patch_size_L)
        RegionNum = RegionNum_col**2
        Region = np.zeros([imgL.shape[0], RegionNum, imgL.shape[1], RegionSize, RegionSize], 'float32')
        for n in range(imgL.shape[0]):
            Region1 = torch.chunk(imgL[n], RegionNum_col, dim=-2)
            jj = 0
            for ii in range(len(Region1)):
                Region2 = torch.chunk(Region1[ii].cpu(), RegionNum_col, dim=-1)
                for cc in range(len(Region2)):
                    Region[n, jj, :, :, :] = Region2[cc]
                    jj += 1
        Region = torch.from_numpy(Region).cuda()

        pL = self.patch_sizeL
        xL = rearrange(imgL, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=pL, p2=pL)

        step = 2
        Dilated_xL = xL[:, ::step, :]

        xL = self.patch_to_embeddingL(Dilated_xL)

        bL, nL, _ = xL.shape

        cls_tokensL = repeat(self.cls_tokenL, '() n d -> b n d', b=bL)
        xL = torch.cat((cls_tokensL, xL), dim=1)

        xL += self.pos_embeddingL[:, :(nL + 1)]
        xL = self.dropoutL(xL)

        context = Region[:, int(RegionNum/2), :, :, :] 
        context = rearrange(context, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2*half_S+1, p2=2*half_S+1)
        context = self.context_to_embedding(context)
        del Region

        xL = self.transformerL(context, xL, mask)

        xL = self.to_cls_tokenL(xL[:, 0])

        xL = self.mlp_headL(xL)

        pS = self.patch_sizeS

        xS = rearrange(imgS, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=pS, p2=pS)
        xS = self.patch_to_embeddingS(xS)
        bS, nS, _ = xS.shape

        cls_tokensS = repeat(self.cls_tokenS, '() n d -> b n d', b=bS)
        xS = torch.cat((cls_tokensS, xS), dim=1)
        xS += self.pos_embeddingS[:, :(nS + 1)]
        xS = self.dropoutS(xS)

        xS = self.transformerS(xS, mask)

        xS = self.to_cls_tokenS(xS[:, 0])

        xS = self.mlp_headS(xS)

        x = xL + xS

        return self.mlp_head(x)




#################################### load dataset ###################################
a=load()

dataID = 5
CLASSES_NUM = 9  # CLASSES_NUM = categories - 1

StorageLocation = './CSIL/Result/1/1_50trn/'
if not os.path.isdir(StorageLocation):
    os.makedirs(StorageLocation)

# Number of experiments
Experiment_num = 1
if dataID == 1:
    All_data, labeled_data, rows_num, categories, r, c, FLAG = a.load_data(flag='pavia')
    epoch = 45
    numPCA = 5
elif dataID == 2:
    All_data, labeled_data, rows_num, categories, r, c, FLAG = a.load_data(flag='indian')
    epoch = 15
    numPCA = 8
elif dataID == 6:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='ksc')
    epoch = 70
    numPCA = 8
elif dataID == 7:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='houston')
    epoch = 75
    numPCA = 3
elif dataID == 4:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='sali')
    epoch = 50
    numPCA = 8
elif dataID == 8:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='hanchuan')
    epoch = 45
    numPCA = 3
elif dataID == 9:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='honghu')
    epoch = 50
    numPCA = 3
elif dataID == 10:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='longkou')
    epoch = 50
    numPCA = 3
elif dataID == 11:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='HU2018')
    epoch = 80
    numPCA = 3
elif dataID == 5:
    All_data,labeled_data,rows_num,categories,r,c,FLAG = a.load_data(flag='paviaC')
    epoch = 45
    numPCA = 5

print('Data has been loaded successfully!')



mi = -0.5
ma = 0.5

a=preprocess('pca')  # ICA or PCA
Alldata_DR = a.Dim_reduction(All_data, numPCA)

a = product(c, FLAG)

Alldata_DR_norm=a.normlization(Alldata_DR,mi,ma)

image_data3D_DR=Alldata_DR_norm.reshape(r,c,-1)



neighborSize = 27
half_L = int(neighborSize * 5 / 2)
patch_size_L = neighborSize
half_S = int(neighborSize / 2)
patch_size_S = 1


image_3ddr_lr=np.fliplr(image_data3D_DR)
image_3ddr_ud=np.flipud(image_data3D_DR)
image_3ddr_corner=np.fliplr(np.flipud(image_data3D_DR))

image_3ddr_temp1=np.hstack((image_3ddr_corner,image_3ddr_ud,image_3ddr_corner))
image_3ddr_temp2=np.hstack((image_3ddr_lr,image_data3D_DR,image_3ddr_lr))
image_3ddr_merge=np.vstack((image_3ddr_temp1,image_3ddr_temp2,image_3ddr_temp1))

image_3ddr_mat_origin_L=image_3ddr_merge[(r-half_L):2*r+half_L,(c-half_L):2*c+half_L,:]

image_3ddr_mat_origin_S=image_3ddr_merge[(r-half_S):2*r+half_S,(c-half_S):2*c+half_S,:]


del image_3ddr_lr,image_3ddr_ud,image_3ddr_corner,image_3ddr_temp1,image_3ddr_temp2,image_3ddr_merge


#################################### 空间数据，训练、检验、测试、预测 ###########################
Experiment_result=np.zeros([categories+4, Experiment_num + 2])

for count in range(0, Experiment_num):
    a = product(c, FLAG)
    rows_num,trn_num,tes_num,pre_num=a.generation_num(labeled_data,rows_num,All_data)
    y_trn = All_data[trn_num, -1]
    trn_YY = torch.from_numpy(y_trn - 1)
    trn_spat_L, trn_spat_S, trn_num, _ = a.production_data_trn(rows_num, trn_num, half_L, image_3ddr_mat_origin_L, half_S, image_3ddr_mat_origin_S)

    np.save(StorageLocation + repr(dataID) + '_trn_num' + '.npy', trn_num)
    np.save(StorageLocation + repr(dataID) + '_pre_num' + '.npy', pre_num)
    np.save(StorageLocation + repr(dataID) + '_y_trn' + '.npy', y_trn)
    np.save(StorageLocation + repr(dataID) + '_image_3d_mat_origin_L' + '.npy', image_3ddr_mat_origin_L)
    np.save(StorageLocation + repr(dataID) + '_image_3d_mat_origin_S' + '.npy', image_3ddr_mat_origin_S)


    trn_XX_spat_L = torch.from_numpy(trn_spat_L.transpose(0, 3, 1, 2))  # (N,C,Depth,H,W)
    trn_XX_spat_S = torch.from_numpy(trn_spat_S.transpose(0, 3, 1, 2))  # (N,C,Depth,H,W)


    torch.cuda.empty_cache()  # GPU memory released

    trn_dataset_L = TensorDataset(trn_XX_spat_L, trn_YY)
    trn_loader_L = DataLoader(trn_dataset_L, batch_size=4
                , sampler=SubsetRandomSampler(range(trn_XX_spat_L.shape[0])))

    trn_dataset_S = TensorDataset(trn_XX_spat_S, trn_YY)
    trn_loader_S = DataLoader(trn_dataset_S, batch_size=4
                , sampler=SubsetRandomSampler(range(trn_XX_spat_S.shape[0])))

    trn_dataset = TensorDataset(trn_XX_spat_L, trn_XX_spat_S, trn_YY)
    trn_loader = DataLoader(trn_dataset, batch_size=4
                , sampler=SubsetRandomSampler(range(trn_XX_spat_S.shape[0])))  # only 'trn' can have 'sampler'


    net = CSIL(
    image_sizeL=2*half_L+1,
    patch_sizeL=patch_size_L,
    image_sizeS=2*half_S+1,
    patch_sizeS=patch_size_S,
    num_classes=CLASSES_NUM,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    channels=numPCA,  # with PCA
    dropout=0.1,
    emb_dropout=0.1).cuda()


    net = net.to(device)
    print("training on ", device, '\n')
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.cuda()
    torch.set_num_threads(1)

    criterion = torch.nn.CrossEntropyLoss()  # 负对数似然损失函数（如果未算log_softmax则直接采用交叉熵损失函数)，否则loss负值
    optimizer=torch.optim.Adam(net.parameters(), lr=1e-3/10, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    loss_trn = []

    trn_time1 = time.time()
    for i in range(1, epoch+1):
        b = operate()
        loss_trn = b.train(i, loss_trn, net, optimizer, scheduler, trn_loader, trn_loader_L, trn_loader_S, criterion)
    trn_time2 = time.time()

    torch.save(net, StorageLocation + 'Exp_' + str(FLAG) + '.pkl')

    print('第{}次实验，模型训练阶段完成！'.format(count))


    ######################################### 分块测试 ##########################################
    y_disp=np.zeros([All_data.shape[0]])
    y_disp[trn_num]=y_trn
    y_pred_tes = y_disp.copy()

    start = 0

    end = np.min([start + 100, len(tes_num)])
    part_num = int(len(tes_num) / 100) + 1
    print('需要分成{}块来测试'.format(part_num))

    tsne_tes = []
    tsne_tes_y = []

    tes_time1 = time.time()

    for i in range(0, part_num):
        ###################### label ######################
        tes_num_part = tes_num[start:end]

        y_tes = All_data[tes_num_part, -1]  # label
        tes_YY = torch.from_numpy(y_tes - 1)  # (100,)

        ######################  data ######################
        a = product(c, FLAG)

        tes_spat_L, tes_spat_S, tes_num_part = a.production_data_valtespre(tes_num_part, half_L, image_3ddr_mat_origin_L, half_S, image_3ddr_mat_origin_S, flag='Tes')

        tes_XX_spat_L = torch.from_numpy(tes_spat_L.transpose(0, 3, 1, 2))  # (N,C,Depth,H,W)
        tes_XX_spat_S = torch.from_numpy(tes_spat_S.transpose(0, 3, 1, 2))  # (N,C,Depth,H,W)

        torch.cuda.empty_cache()

        tes_dataset_L = TensorDataset(tes_XX_spat_L, tes_YY)
        tes_loader_L = DataLoader(tes_dataset_L, batch_size=4)

        tes_dataset_S = TensorDataset(tes_XX_spat_S, tes_YY)
        tes_loader_S = DataLoader(tes_dataset_S, batch_size=4)

        tes_dataset = TensorDataset(tes_XX_spat_L, tes_XX_spat_S, tes_YY)
        tes_loader_part = DataLoader(tes_dataset, batch_size=4)

        del tes_dataset_L, tes_dataset_S

        net = torch.load(StorageLocation + 'Exp_' + str(FLAG) + '.pkl', map_location='cpu')

        net = net.cuda()
        torch.set_num_threads(1)

        a = operate()

        y_pred_tes_part = a.inference(net, tes_loader_part, criterion, FLAG='TEST')

        y_pred_tes[tes_num_part] = y_pred_tes_part

        start = end
        end = np.min([start + 100, len(tes_num)])

    tes_time2 = time.time()


    ###################### Assess, 测试集 #######################
    print('==================Test set=====================')
    y_tes = All_data[tes_num, -1]  # label
    tes_YY = torch.from_numpy(y_tes - 1)
    y_tes = tes_YY.numpy() + 1
    print('第{}次实验，测试集OA={}'.format(count,np.mean(y_tes==y_pred_tes[tes_num])))
    print('第{}次实验，测试集Kappa={}'.format(count,cohen_kappa_score(y_tes,y_pred_tes[tes_num])))
    ########## 各类别精度 ##########
    num_tes = np.zeros([categories - 1])
    num_tes_pred = np.zeros([categories - 1])

    for k in y_tes:
        num_tes[k - 1] = num_tes[k - 1] + 1

    for j in range(y_tes.shape[0]):
        if y_tes[j] == y_pred_tes[tes_num][j]:
            num_tes_pred[y_tes[j] - 1] = num_tes_pred[y_tes[j] - 1] + 1

    Acc = num_tes_pred / num_tes * 100

    Experiment_result[0, count] = np.mean(y_tes == y_pred_tes[tes_num]) * 100  # OA
    Experiment_result[1, count] = np.mean(Acc)  # AA
    Experiment_result[2, count] = cohen_kappa_score(y_tes, y_pred_tes[tes_num]) * 100  # Kappa
    Experiment_result[3, count] = trn_time2 - trn_time1
    Experiment_result[4, count] = tes_time2 - tes_time1
    Experiment_result[5:, count] = Acc

    ############################ Output Excel #############################
    # 准备数据
    data_df = pd.DataFrame(Experiment_result)  # 将ndarray格式转换为DataFrame
    # 将文件写入excel表格
    writer = pd.ExcelWriter(StorageLocation + str(FLAG) + '_' + str(int(Experiment_result[0, 0]*100)) + '.xls')  # 创建excel表格
    data_df.to_excel(writer, 'Acc&Time')  # float_format 控制精度，将data_df写到表格第一页。若多个文件，可以在page_2写入
    writer.save()

    print('第{}次实验，模型评估阶段完成！'.format(count))


    ############################### 分块出图 ##############################
    y_disp=np.zeros([All_data.shape[0]])
    y_disp[trn_num]=y_trn
    y_pred = y_disp.copy()

    start = 0

    end = np.min([start + 100, len(pre_num)])
    part_num = int(len(pre_num) / 100) + 1
    print('需要分成{}块来出图'.format(part_num))

    for i in range(0, part_num):
        pre_num_part = pre_num[start:end]

        y_pre = All_data[pre_num_part, -1]
        pre_YY = torch.from_numpy(np.ones([y_pre.shape[0]]))

        a = product(c, FLAG)

        pre_spat_L, pre_spat_S, pre_num_part = a.production_data_valtespre(pre_num_part, half_L, image_3ddr_mat_origin_L, half_S, image_3ddr_mat_origin_S, flag='Pre')

        pre_XX_spat_L = torch.from_numpy(pre_spat_L.transpose(0, 3, 1, 2))
        pre_XX_spat_S = torch.from_numpy(pre_spat_S.transpose(0, 3, 1, 2))

        torch.cuda.empty_cache()

        pre_dataset = TensorDataset(pre_XX_spat_L, pre_XX_spat_S, pre_YY)
        pre_loader_part = DataLoader(pre_dataset, batch_size=100)

        net = torch.load(StorageLocation + 'Exp_' + str(FLAG) + '.pkl', map_location='cpu')

        net = net.cuda()
        torch.set_num_threads(1)

        a = operate()

        y_pred_part = a.inference(net, pre_loader_part, criterion, FLAG='PRED')

        y_pred[pre_num_part] = y_pred_part

        start = end
        end = np.min([start + 100, len(pre_num)])

    y_disp=np.zeros([All_data.shape[0]])
    y_disp[trn_num]=y_trn
    y_disp[tes_num]=y_pred_tes[tes_num]

    y_disp_all=y_disp.copy()
    y_disp_all[pre_num]=y_pred[pre_num]

    X_result = DrawResult(y_disp_all.astype(int), dataID)
    plt.imsave(StorageLocation + str(FLAG) + '_' + str(int(np.mean(y_tes == y_pred_tes[tes_num])*10000)) + '.bmp', X_result)
    print('结果图保存阶段完成！', '\n')


######### 计算多次实验的均值与标准差并保存 #############
Experiment_result[:,-2]=np.mean(Experiment_result[:,0:-2],axis=1)
Experiment_result[:,-1]=np.std(Experiment_result[:,0:-2],axis=1)
scio.savemat(StorageLocation + str(FLAG) + '_AllResults' + repr(int(Experiment_result[0, -2]*100)) + '.mat',{'data':Experiment_result})
######### Output Mean&Std to Excel #######
data_df = pd.DataFrame(Experiment_result)
writer = pd.ExcelWriter(
    StorageLocation + str(FLAG) + '_AllMeanStd_' + repr(int(Experiment_result[0, -2]*100)) + '.xls')
data_df.to_excel(writer, 'All&Mean&Std')
writer.save()

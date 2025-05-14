import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from torch.nn import functional as F

####################################################################
class CNNEncoder(nn.Module):
    def __init__(self,input_channels,feature_dim=64):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(input_channels, feature_dim, kernel_size=1, padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())
        self.layer2 = nn.Sequential(
                        nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())
        self.layer3 = nn.Sequential(
                        nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())
        self.pool = nn.MaxPool2d(2)

    def forward(self,x):
        result1 = self.layer1(x)
        if result1.shape[3]<8:
            pass
        else:
            result1 = self.pool(result1)
        result2 = self.layer2(result1)
        result3 = self.layer3(result2)
        result4 = self.layer4(result3)
        return result2,result3,result4
####################################################################
class Attention1(nn.Module):

    def __init__(self, channel=512,reduction=16,feature_dim=5):
        super().__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel*2, channel*2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel*2 // reduction, channel*2, bias=False),
            nn.Sigmoid()
        )
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.avg_pool3 = nn.AdaptiveAvgPool2d(1)
        self.fc3 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim * 2, kernel_size=1, padding=0),
            nn.BatchNorm2d(feature_dim * 2, momentum=1, affine=True),
            nn.ReLU())
        self.w = nn.Parameter(torch.tensor([1.0,1.0,1.0]),requires_grad=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x1, x2, x3, run=1500):
        b1, c1, _, _ = x1.size()
        y1 = self.avg_pool1(x1).view(b1, c1)
        y1 = self.fc1(y1).view(b1, c1, 1, 1)
        q1 = (x1 * y1).expand_as(x1)

        b2, c2, _, _ = x2.size()
        y2 = self.avg_pool2(x2).view(b2, c2)
        y2 = self.fc2(y2).view(b2, c2, 1, 1)
        q2 = (x2 * y2).expand_as(x2)

        b3, c3, _, _ = x3.size()
        y3 = self.avg_pool3(x3).view(b3, c3)
        y3 = self.fc3(y3).view(b3, c3, 1, 1)
        q3 = (x3 * y3).expand_as(x3)

        w_concat = self.w[0]/torch.sum(self.w)
        w_product = self.w[1]/torch.sum(self.w)
        w_difference = self.w[2]/torch.sum(self.w)

        q = torch.cat((w_concat*q1, w_product*q2, w_difference*q3),1)
        q = self.conv(q)
        return q
####################################################################
class RelationNetwork1(nn.Module):
    def __init__(self, patch_size,feature_dim=64,k=5):
        super(RelationNetwork1, self).__init__()
        if patch_size<8:
            pass
        else:
            patch_size = patch_size//2
        self.remain = patch_size*patch_size-4
        self.row = (patch_size*patch_size)//k
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,feature_dim,kernel_size=(1,self.row),padding=0,stride=(1,self.row)),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())
        self.layer2 = nn.Sequential(
                        nn.Conv2d(feature_dim,feature_dim,kernel_size=3,padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())
        self.layer3 = nn.Sequential(
                        nn.Conv2d(feature_dim, 1, kernel_size=3, padding=0),
                        nn.BatchNorm2d(1, momentum=1, affine=True),
                        nn.ReLU())
        self.eca = ECA_Module(feature_dim)

    def forward(self,x):
        out = self.layer1(x)
        out = self.eca(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = torch.sigmoid(out)
        return out
####################################################################
class RelationNetwork2(nn.Module):
    def __init__(self, patch_size,feature_dim=64,k=5):
            super(RelationNetwork2, self).__init__()
            if patch_size < 8:
                pass
            else:
                patch_size = patch_size // 2
            self.remain = patch_size*patch_size-4
            self.row = (patch_size*patch_size)//k
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, feature_dim, kernel_size=(1, self.row), padding=0, stride=(1, self.row)),
                nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                nn.ReLU())
            self.layer2 = nn.Sequential(
                            nn.Conv2d(feature_dim,feature_dim,kernel_size=3,padding=0),
                            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                            nn.ReLU())
            self.layer3 = nn.Sequential(
                nn.Conv2d(feature_dim, 1, kernel_size=3, padding=0),
                nn.BatchNorm2d(1, momentum=1, affine=True),
                nn.ReLU())
            self.eca = ECA_Module(feature_dim)

    def forward(self,x):
            out = self.layer1(x)
            out = self.eca(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = out.view(out.size(0), -1)
            out = torch.sigmoid(out)
            return out
####################################################################
class RelationNetwork3(nn.Module):
    def __init__(self, patch_size,feature_dim=64):
            super(RelationNetwork3, self).__init__()
            self.layer1 = nn.Sequential(
                            nn.Conv2d(feature_dim*2,feature_dim,kernel_size=1,padding=0),
                            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                            nn.ReLU())
            self.layer2 = nn.Sequential(
                            nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                            nn.ReLU())
            if patch_size<8:
                pass
            else:
                patch_size = patch_size//2
            self.layer3 = nn.Conv2d(feature_dim,1,kernel_size=patch_size,padding=0)
    def forward(self,x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0),-1)
            out = torch.sigmoid(out)
            return out
####################################################################
class RelationNetwork3_1(nn.Module):
    def __init__(self, patch_size,feature_dim=64):
            super(RelationNetwork3_1, self).__init__()
            self.layer1 = nn.Sequential(
                            nn.Conv2d(feature_dim*2,feature_dim,kernel_size=1,padding=0),
                            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                            nn.ReLU())
            self.layer2 = nn.Sequential(
                            nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                            nn.ReLU())
            if patch_size<8:
                pass
            else:
                patch_size = patch_size//2
            self.layer3 = nn.Conv2d(feature_dim,1,kernel_size=patch_size,padding=0)
    def forward(self,x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0),-1)
            out = torch.sigmoid(out)
            return out
####################################################################
class RelationNetwork3_2(nn.Module):
    def __init__(self, patch_size,feature_dim=64):
            super(RelationNetwork3_2, self).__init__()
            self.layer1 = nn.Sequential(
                            nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                            nn.ReLU())
            self.layer2 = nn.Sequential(
                            nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                            nn.ReLU())
            if patch_size<8:
                pass
            else:
                patch_size = patch_size//2
            self.layer3 = nn.Conv2d(feature_dim,1,kernel_size=patch_size,padding=0)
    def forward(self,x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0),-1)
            out = torch.sigmoid(out)
            return out
####################################################################
class RelationNetwork3_3(nn.Module):
    def __init__(self, patch_size,feature_dim=64):
            super(RelationNetwork3_3, self).__init__()
            self.layer1 = nn.Sequential(
                            nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                            nn.ReLU())
            self.layer2 = nn.Sequential(
                            nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                            nn.ReLU())
            if patch_size<8:
                pass
            else:
                patch_size = patch_size//2
            self.layer3 = nn.Conv2d(feature_dim,1,kernel_size=patch_size,padding=0)
    def forward(self,x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0),-1)
            out = torch.sigmoid(out)
            return out
####################################################################
class imagetoclass2(nn.Module):
    def __init__(self, BATCH_SIZE_PER_CLASS, N_class, feature_dim, SAMPLE_SIZE):
        super(imagetoclass2, self).__init__()
        self.batch_size_per_class = BATCH_SIZE_PER_CLASS
        self.n_class = N_class
        self.feature_dim = feature_dim
        self.sample_size = SAMPLE_SIZE

    def cal_L(self, S, Q, t_index, s_list, mode, k):
        task_index = t_index.copy()
        special_list = s_list.copy()
        count = 0
        S_add_T_norm = []
        for i in range(self.n_class):
            Si = []
            if i+1 in task_index:
                if mode == 'train':
                    for j in range(special_list[0]//2):
                        Si.append(S[count])
                        count = count + 1
                    del special_list[0]
                elif mode == 'test':
                    for j in range(special_list[0]):
                        Si.append(S[count])
                        count = count + 1
                    del special_list[0]
            else:
                if mode == 'train':
                    for j in range(self.batch_size_per_class):
                        Si.append(S[count])
                        count = count + 1
                elif mode == 'test':
                    for j in range(self.sample_size):
                        Si.append(S[count])
                        count = count + 1
            Si = torch.cat(Si, dim=1).unsqueeze(0)
            Si = Si.view(Si.size(0), Si.size(1), -1)
            Si = torch.transpose(Si, 1, 2)
            support_set_sam = Si[0]
            support_set_sam_norm = torch.norm(support_set_sam, 2, 1, True)
            support_set_sam = support_set_sam / support_set_sam_norm
            S_add_T_norm.append(support_set_sam.unsqueeze(0))

        Q = Q.view(Q.size(0), Q.size(1), -1)
        Q_norm = []
        for i in range(Q.size(0)):
            query_set_sam = Q[i]
            query_set_sam_norm = torch.norm(query_set_sam, 2, 0, True)
            query_set_sam = query_set_sam / query_set_sam_norm
            Q_norm.append(query_set_sam.unsqueeze(0))

        b = []
        for i in range(len(S_add_T_norm)):
            Si = S_add_T_norm[i]
            a = []
            for j in range(len(Q_norm)):
                Qj = Q_norm[j]
                L = Si @ Qj
                L = L[0]
                topk_value, _ = torch.topk(L, k, 0)
                a.append(topk_value.unsqueeze(0))
            a = torch.cat(a,dim=0)
            b.append(a.unsqueeze(0))
        b = torch.cat(b,dim=0)

        b = b.unsqueeze(2)
        return b

    def forward(self, support, query, task_index,special_list,mode,k):
        _, _, h2, w2 = query.size()
        L = self.cal_L(support, query,task_index,special_list, mode, k).view(-1, 1, k, h2 * w2)
        return L
####################################################################
class relation_weight(nn.Module):
    def __init__(self):
        super(relation_weight, self).__init__()
        self.w = nn.Parameter(torch.tensor([1.0,1.0,1.0]),requires_grad=True)

    def forward(self,x1,x2,x3,run=2000):
        x = (self.w[0] * x1+self.w[1] * x2+self.w[2] * x3)/(self.w[0]+self.w[1]+self.w[2])
        return x
####################################################################
class ECA_Module(nn.Module):
    def __init__(self, channel,gamma=2, b=1):
        super(ECA_Module, self).__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(channel, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        y = y.transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
####################################################################
def cal_loss(criterion_1,result_input,one_hot_labels_input):
    loss1 = criterion_1(result_input[0], one_hot_labels_input)
    loss2 = criterion_1(result_input[1], one_hot_labels_input)
    loss3 = criterion_1(result_input[2], one_hot_labels_input)
    loss3_1 = criterion_1(result_input[3], one_hot_labels_input)
    loss3_2 = criterion_1(result_input[4], one_hot_labels_input)
    loss3_3 = criterion_1(result_input[5], one_hot_labels_input)
    loss4 = criterion_1(result_input[6], one_hot_labels_input)
    l_stage = loss1 + loss2 + loss3
    l_fr = loss3_1 + loss3_2 + loss3_3
    l_fused = loss4
    return l_fr,l_stage,l_fused
####################################################################
class DM_MRN_train(nn.Module):
    def __init__(self,N_BANDS,FEATURE_DIM,PATCH_SIZE,numComponents,BATCH_SIZE_PER_CLASS, N_CLASSES, SAMPLE_SIZE,k):
        super(DM_MRN_train, self).__init__()
        self.FEATURE_DIM = FEATURE_DIM
        self.BATCH_SIZE_PER_CLASS = BATCH_SIZE_PER_CLASS
        self.N_CLASSES = N_CLASSES
        self.SAMPLE_SIZE = SAMPLE_SIZE
        self.k = k
        if numComponents=='all' or numComponents=='without':
            self.encoder = CNNEncoder(N_BANDS,self.FEATURE_DIM)
        else:
            self.encoder = CNNEncoder(numComponents, self.FEATURE_DIM)
        self.relation1 = RelationNetwork1(PATCH_SIZE,self.FEATURE_DIM, k=self.k)
        self.relation2 = RelationNetwork2(PATCH_SIZE,self.FEATURE_DIM, k=self.k)
        self.relation3 = RelationNetwork3(PATCH_SIZE,self.FEATURE_DIM)
        self.relation3_1 = RelationNetwork3_1(PATCH_SIZE,self.FEATURE_DIM)
        self.relation3_2 = RelationNetwork3_2(PATCH_SIZE,self.FEATURE_DIM)
        self.relation3_3 = RelationNetwork3_3(PATCH_SIZE,self.FEATURE_DIM)
        self.mutil_block = Attention1(channel=self.FEATURE_DIM,reduction=16,feature_dim=self.FEATURE_DIM)
        self.imagetoclass2 = imagetoclass2(self.BATCH_SIZE_PER_CLASS, self.N_CLASSES, self.FEATURE_DIM, self.SAMPLE_SIZE)
        self.sum = relation_weight()

    def forward(self,support,query,task_index,special_list,iters):
        support = self.encoder(support)
        query = self.encoder(query)

        L1 = self.imagetoclass2(support[0],query[0],task_index,special_list,'train',k=self.k)
        L1 = self.relation1(L1)
        L1 = L1.view(-1, self.N_CLASSES)
        relation1_copy = []
        c = 0
        for i in range(self.N_CLASSES):
            if i+1 in task_index:
                for j in range(special_list[c]//2):
                    relation1_copy.append(L1[i].unsqueeze(0))
                c = c + 1
            else:
                for j in range(self.BATCH_SIZE_PER_CLASS):
                    relation1_copy.append(L1[i].unsqueeze(0))
        relation1_copy = torch.cat(relation1_copy, dim=0)

        L1 = self.imagetoclass2(support[1], query[1], task_index, special_list, 'train',k=self.k)
        L1 = self.relation2(L1)
        L1 = L1.view(-1, self.N_CLASSES)
        relation2_copy = []
        c = 0
        for i in range(self.N_CLASSES):
            if i + 1 in task_index:
                for j in range(special_list[c] // 2):
                    relation2_copy.append(L1[i].unsqueeze(0))
                c = c + 1
            else:
                for j in range(self.BATCH_SIZE_PER_CLASS):
                    relation2_copy.append(L1[i].unsqueeze(0))
        relation2_copy = torch.cat(relation2_copy, dim=0)

        query_ext = query[2].unsqueeze(0).repeat(support[2].size(0), 1, 1, 1, 1)
        support_ext = support[2].unsqueeze(0).repeat(query[2].size(0), 1, 1, 1, 1)
        support_ext = torch.transpose(support_ext, 0, 1)
        relation_concat = torch.cat((query_ext, support_ext), 2)
        relation_concat = relation_concat.view(-1, self.FEATURE_DIM * 2, relation_concat.shape[3], relation_concat.shape[4])
        relation_product = query_ext * support_ext
        relation_product = relation_product.view(-1, self.FEATURE_DIM, relation_product.shape[3], relation_product.shape[4])
        relation_difference = torch.abs(query_ext - support_ext)
        relation_difference = relation_difference.view(-1, self.FEATURE_DIM, relation_difference.shape[3], relation_difference.shape[4])
        L1 = self.mutil_block(relation_concat, relation_product, relation_difference, iters)
        relation3 = self.relation3(L1)
        relation3 = relation3.view(-1, self.N_CLASSES)
        relation3_1 = self.relation3_1(relation_concat)
        relation3_1 = relation3_1.view(-1, self.N_CLASSES)
        relation3_2 = self.relation3_2(relation_product)
        relation3_2 = relation3_2.view(-1, self.N_CLASSES)
        relation3_3 = self.relation3_3(relation_difference)
        relation3_3 = relation3_3.view(-1, self.N_CLASSES)

        relation = self.sum(relation1_copy,relation2_copy,relation3,run=iters)

        return relation1_copy,relation2_copy,relation3,relation3_1,relation3_2,relation3_3,relation
####################################################################
class DM_MRN_test(nn.Module):
    def __init__(self, N_BANDS, FEATURE_DIM, PATCH_SIZE,numComponents,BATCH_SIZE_PER_CLASS, N_CLASSES, SAMPLE_SIZE,k):
        super(DM_MRN_test, self).__init__()
        self.FEATURE_DIM = FEATURE_DIM
        self.BATCH_SIZE_PER_CLASS = BATCH_SIZE_PER_CLASS
        self.N_CLASSES = N_CLASSES
        self.SAMPLE_SIZE = SAMPLE_SIZE
        self.k = k
        if numComponents == 'all' or numComponents == 'without':
            self.encoder = CNNEncoder(N_BANDS, self.FEATURE_DIM)
        else:
            self.encoder = CNNEncoder(numComponents, self.FEATURE_DIM)
        self.relation1 = RelationNetwork1(PATCH_SIZE, self.FEATURE_DIM, k=self.k)
        self.relation2 = RelationNetwork2(PATCH_SIZE, self.FEATURE_DIM, k=self.k)
        self.relation3 = RelationNetwork3(PATCH_SIZE, self.FEATURE_DIM)
        self.relation3_1 = RelationNetwork3_1(PATCH_SIZE, self.FEATURE_DIM)
        self.relation3_2 = RelationNetwork3_2(PATCH_SIZE, self.FEATURE_DIM)
        self.relation3_3 = RelationNetwork3_3(PATCH_SIZE, self.FEATURE_DIM)
        self.mutil_block = Attention1(channel=self.FEATURE_DIM, reduction=16, feature_dim=self.FEATURE_DIM)
        self.imagetoclass2 = imagetoclass2(self.BATCH_SIZE_PER_CLASS, self.N_CLASSES, self.FEATURE_DIM, self.SAMPLE_SIZE)
        self.sum = relation_weight()

    def forward(self, train, test, task_index, special_list, N_Test):
        train = self.encoder(train)
        test = self.encoder(test)

        L1 = self.imagetoclass2(train[0], test[0], task_index, special_list, 'test', k=self.k)
        L1 = self.relation1(L1)
        L1 = L1.view(-1, N_Test)
        relation1_copy = []
        c = 0
        for i in range(self.N_CLASSES):
            if i + 1 in task_index:
                for j in range(special_list[c]):
                    relation1_copy.append(L1[i].unsqueeze(0))
                c = c + 1
            else:
                for j in range(self.SAMPLE_SIZE):
                    relation1_copy.append(L1[i].unsqueeze(0))
        relation1_copy = torch.cat(relation1_copy, dim=0)

        L1 = self.imagetoclass2(train[1], test[1], task_index, special_list, 'test', k=self.k)
        L1 = self.relation2(L1)
        L1 = L1.view(-1, N_Test)
        relation2_copy = []
        c = 0
        for i in range(self.N_CLASSES):
            if i + 1 in task_index:
                for j in range(special_list[c]):
                    relation2_copy.append(L1[i].unsqueeze(0))
                c = c + 1
            else:
                for j in range(self.SAMPLE_SIZE):
                    relation2_copy.append(L1[i].unsqueeze(0))
        relation2_copy = torch.cat(relation2_copy, dim=0)

        te_ext = test[2].unsqueeze(0).repeat(train[2].size(0), 1, 1, 1, 1)
        tr_ext = train[2].unsqueeze(0).repeat(test[2].size(0), 1, 1, 1, 1)
        tr_ext = torch.transpose(tr_ext, 0, 1)
        trte_concat = torch.cat((tr_ext, te_ext), 2).view(-1, self.FEATURE_DIM * 2, tr_ext.shape[3], tr_ext.shape[4])
        trte_product = tr_ext * te_ext
        trte_product = trte_product.contiguous().view(-1, self.FEATURE_DIM, trte_product.shape[3], trte_product.shape[4])
        trte_difference = torch.abs(tr_ext - te_ext)
        trte_difference = trte_difference.contiguous().view(-1, self.FEATURE_DIM, trte_difference.shape[3],
                                                            trte_difference.shape[4])
        pairs = self.mutil_block(trte_concat, trte_product, trte_difference)
        relation3 = self.relation3(pairs)
        relation3 = relation3.view(-1, N_Test)

        relation = self.sum(relation1_copy, relation2_copy, relation3)

        return relation
####################################################################

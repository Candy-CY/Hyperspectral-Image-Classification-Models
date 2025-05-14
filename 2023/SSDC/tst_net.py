import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import SAGEConv
from OT_torch_ import cost_matrix_batch_torch, GW_distance_uniform, IPOT_distance_torch_batch_uniform
import math
from torch_geometric.data import Data
import numpy as np

CLASS_NUM = 16

def getGraphdata(source_share, bs, target_share, target=True):
    if (bs == 27):
        segments = torch.reshape(torch.tensor(range(bs)),(3,9))
    else: 
        segments = torch.reshape(torch.tensor(range(bs)),(-1,int(math.sqrt(bs))))
    src_edge = torch.tensor(getEdge(source_share, segments)).t().contiguous()
    source_share_graph = Data(x=source_share,edge_index=src_edge).cuda()
    if target == True:
        tar_edge = torch.tensor(getEdge(target_share, segments)).t().contiguous()
        target_share_graph = Data(x=target_share,edge_index=tar_edge).cuda()
    else:
        target_share_graph =  0
    return source_share_graph, target_share_graph

def getEdge(image, segments, compactness=300, sigma=3.):
    coo = set()
    dire = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
    for i in range(1, segments.shape[0]):
        for j in range(1, segments.shape[1]):
            for dx, dy in dire:
                if -1 < i + dx < segments.shape[0] and \
                        -1 < j + dy < segments.shape[1] and \
                        segments[i, j] != segments[i + dx, j + dy]:
                    coo.add((segments[i, j], segments[i + dx, j + dy]))

    coo = np.asarray(list(coo))
    return coo

class Topology_Extraction(torch.nn.Module):
    def __init__(self, in_channels,num_classes,dropout=0.5):
        super(Topology_Extraction, self).__init__()
        self.conv1 = SAGEConv(in_channels, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = SAGEConv(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.mlp_classifier = nn.Sequential(
            nn.Linear(32, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes, bias=True)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x_temp_1 = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x_temp_2 = x
        x = self.mlp_classifier(x)
        return F.softmax(x, dim=1), x, x_temp_1, x_temp_2


class Feature_Extractor(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super(Feature_Extractor, self).__init__()
        self.classes = num_classes
        self.gcn = Global_graph(1440,CLASS_NUM)

    def forward(self, source_feature, target_feature):
        bs = source_feature.shape[0]
        source_share_graph, target_share_graph = getGraphdata(source_feature, bs, target_feature)
        source_gcn_pred, _, TST_wd, TST_gwd = self.gcn(source_share_graph,target_share_graph)
        return TST_wd, TST_gwd, source_gcn_pred


class Global_graph(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Global_graph, self).__init__()
        self.sharedNet_src = Topology_Extraction(in_channels,num_classes)
        self.sharedNet_tar = Topology_Extraction(in_channels,num_classes)

    def forward(self, source, target):
        wd_ori, gwd_ori = OT(source, target, ori=True)
        out = self.sharedNet_src(source)
        p_source, source, source_share_1, source_share_2 = out[0], out[1], out[2], out[3]
        out = self.sharedNet_tar(target)
        p_target, target, target_share_1, target_share_2 = out[0], out[1], out[2], out[3]
        wd_1, gwd_1 = OT(source_share_1, target_share_1)
        wd_2, gwd_2 = OT(source_share_2, target_share_2)
        wd = wd_ori + wd_1 + wd_2
        gwd = gwd_ori + gwd_1 + gwd_2
        return source, target, wd, gwd

def OT(source_share, target_share, ori=False):
    if ori == True:
        source_share = source_share.x.unsqueeze(0).transpose(2,1)
        target_share = target_share.x.unsqueeze(0).transpose(2,1)
    else:
        source_share = source_share.unsqueeze(0).transpose(2,1)
        target_share = target_share.unsqueeze(0).transpose(2,1)
    
    cos_distance = cost_matrix_batch_torch(source_share, target_share)
    cos_distance = cos_distance.transpose(1,2)
    # TODO: GW and Gwd as graph alignment loss
    beta = 0.1
    min_score = cos_distance.min()
    max_score = cos_distance.max()
    threshold = min_score + beta * (max_score - min_score)
    cos_dist = torch.nn.functional.relu(cos_distance - threshold)
    wd = - IPOT_distance_torch_batch_uniform(cos_dist, source_share.size(0), source_share.size(2), target_share.size(2), iteration=30)
    gwd = GW_distance_uniform(source_share, target_share)
    return torch.mean(wd), torch.mean(gwd)

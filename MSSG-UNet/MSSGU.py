import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn, knn_graph
from torch_geometric.utils import add_self_loops, degree,grid, remove_self_loops,dense_to_sparse
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCNLayer_PyG(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, adjacency_matrix:torch.Tensor, position: torch.Tensor,neighbors:int):
        super(GCNLayer_PyG, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition1 = nn.LeakyReLU(inplace=True)

        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 128))
        
        self.position=position
        self.neighbors = neighbors

        self.a = nn.Parameter(torch.ones(size=(1, 1), requires_grad=True, device=device))
        self.b = nn.Parameter(torch.ones(size=(1, 1), requires_grad=True, device=device))
        self.lambda_ = nn.Parameter(torch.zeros(1))
        
        self.theta1 = nn.Sequential(nn.Linear(input_dim, 1))#,nn.Sigmoid()
        if self.neighbors>0:
            self.neighbors=self.neighbors+1 # self-loop
            self.col, self.row = self.edge_index = knn_graph(self.position, self.neighbors, batch=None, loop=True)  # 
        else:
            # unfixed neighbors
            self.I = torch.eye(adjacency_matrix.shape[0], adjacency_matrix.shape[0], requires_grad=False, device=device, dtype=torch.float32)
            self.mask = torch.ceil(adjacency_matrix * 0.00001)
            self.index, _=dense_to_sparse(adjacency_matrix.contiguous()+self.I)
            self.row,self.col= self.index
            
        
        ########################spatial distance##########################
        if self.neighbors>0: self.Spatial_Distance = torch.square(torch.norm(self.position[self.col] - self.position[self.row], dim=-1))
        
    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, prior_A=None):
        H = self.BN(H)
        node_count = H.shape[0]
        H_xx1 = self.GCN_liner_theta_1(H)
        out = self.GCN_liner_out_1(H)
        
        if self.neighbors>0:
            col, row = self.col, self.row
            A=torch.sigmoid(torch.multiply(H_xx1[col],H_xx1[row]).sum(-1))
            A = A.reshape([node_count, self.neighbors, -1])
            A=torch.softmax(A,dim=1)
            out=out[col].reshape([node_count,self.neighbors,-1])
            out = self.Activition1(torch.multiply(A,out).sum(1))
        else:
            ###################### Softmax
            e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
            zero_vec = -9e15 * torch.ones_like(e)
            A = torch.where(self.mask > 0, e, zero_vec)+ self.lambda_*self.I
            A_softmax = F.softmax(A, dim=1)
            out = self.Activition1(torch.mm(A_softmax, out))

        return out,A

class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )
        self.Act1 = nn.LeakyReLU(inplace=True)
        self.Act2 = nn.LeakyReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_ch)
        # self.BN2 = nn.BatchNorm2d(out_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        # out=self.BN2(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out

class HiGCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int,
                 hierarchy_matrices, adjacency_matrices,neighbors):
        super(HiGCN, self).__init__()
        self.class_count = class_count  
        self.channel = changel
        self.height = height
        self.width = width
        self.S_list = hierarchy_matrices
        self.A_list = adjacency_matrices
        self.layer_count=len(hierarchy_matrices) 
        self.neighbors = neighbors
        self.S_list_Hat_T = []
        for i in range(len(hierarchy_matrices)):
            temp=hierarchy_matrices[i]
            self.S_list_Hat_T.append((temp/ (torch.sum(temp, 0, keepdim=True,dtype=torch.float32))).t())  # Q
        
        positions = self.getPositions()
        self.positions=[]
        for i in range(len(hierarchy_matrices)):
            positions = torch.mm(self.S_list_Hat_T[i], positions)
            self.positions.append(positions)
            
            
        layer_channels=128
        layers_per_depth=[layer_channels]
        for i in range(len(hierarchy_matrices)):
            layer_channels=layer_channels//2
            layers_per_depth.append(layer_channels)

        self.CNN_head_layer=SSConv(self.channel, layers_per_depth[0], kernel_size=5)

        self.GCN_layers=nn.Sequential()
        for i in range(self.layer_count):
            self.GCN_layers.add_module('my_GCN_layer'+str(i), GCNLayer_PyG(layers_per_depth[i],layers_per_depth[i+1],
                                                                       self.A_list[i],
                                                                       self.positions[i],self.neighbors))
            
        self.De_GCN_layers=nn.Sequential()
        for i in range(self.layer_count-1):
            self.De_GCN_layers.add_module('my_DeGCN_layer'+str(i), GCNLayer_PyG(layers_per_depth[-i-1]+layers_per_depth[-i-2],layers_per_depth[-i-2],
                                                                                # layers_per_depth[-i - 1], layers_per_depth[-i - 2],
                                                                            self.A_list[-i-2],
                                                                            self.positions[self.layer_count-i-2],self.neighbors))
        
        if len(layers_per_depth)<=1: layers_per_depth.append(layer_channels)
        self.CNN_tail_layer=SSConv(layers_per_depth[0]+layers_per_depth[1],layers_per_depth[0], kernel_size=5)
        self.Softmax = nn.Sequential(nn.Linear(layers_per_depth[0], self.class_count), nn.Softmax(-1))

    def getPositions(self):
        # for KNN
        x=torch.arange(0,self.height)
        y=torch.arange(0,self.width)
        x,y = torch.meshgrid([x,y])
        xy=torch.stack([x,y],-1).reshape([self.height*self.width,2]).float()
        return xy.to(device=device)
        
    
    def forward(self, x: torch.Tensor, showFlag=False):
        '''
        :param x: C*H*W
        :return: probability_map H*W*C
        '''
        (h, w, c) = x.shape
        x=torch.unsqueeze(x.permute([2, 0, 1]), 0)
        H_0 = self.CNN_head_layer(x)
        H_0 = torch.squeeze(H_0, 0).permute([1, 2, 0])
        H_i= H_0.reshape([h * w, -1])
        encoder_features =[]
        encoder_features.append(H_i)
        
        decoder_features = [] #
        A_encoders=[]; A_decoders=[]
        
        # encoder GCN
        for i in range(len(self.GCN_layers)):
            H_i = torch.mm(self.S_list_Hat_T[i], H_i)
            H_i,A_i = self.GCN_layers[i](H_i,prior_A=None) 
            encoder_features.append(H_i)
            A_encoders.append(A_i)
            
        A_decoders.append(A_encoders[-1])# decoder encoder 
        decoder_features.append(H_i)
        
        for i in range(len(self.De_GCN_layers)):
            H_i=torch.mm(self.S_list[-i-1], H_i)
            H_i=torch.cat([H_i,encoder_features[-i-2]],dim=-1)  # concat 
            # H_i=H_i+encoder_features[-i-2]  # add 
            H_i,A_i = self.De_GCN_layers[i]( H_i ,prior_A=None) 
            A_decoders.append(A_i)
            decoder_features.append(H_i)

        if len(self.S_list)>0: H_i=torch.mm(self.S_list[0],H_i) # hw*d
        H_i = torch.cat([H_i, encoder_features[0]],dim=-1) # hw*(2d) concat 模式

        # tail CNN
        cnn_tail_input=H_i.reshape([1,h,w,-1]).permute([0,3,1,2])
        final_features=self.CNN_tail_layer(cnn_tail_input)
        
        # softmax
        final_features=torch.squeeze(final_features, 0).permute([1, 2, 0]).reshape([h * w, -1])
        decoder_features.append(final_features)  
        Y = self.Softmax(final_features)
        
        return Y, encoder_features, decoder_features

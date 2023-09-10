import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift, random_walker
from sklearn import preprocessing
import cv2
import math
from torch_geometric.nn import knn_graph
from torch_geometric.utils import k_hop_subgraph,to_dense_adj,dense_to_sparse

def LSC_superpixel(I, nseg):
    height, width, bands = I.shape  # 原始高光谱数据的三个维度
    I = np.reshape(I, [height * width, bands])
    minMax = preprocessing.MinMaxScaler(feature_range=(0,1))
    I  = minMax.fit_transform(I)
    I = np.reshape(I, [height, width, bands])
    I = np.array(I * 255, dtype='uint8')
    size = int(math.sqrt(((I.shape[0] * I.shape[1]) / nseg)))
    superpixelLSC = cv2.ximgproc.createSuperpixelLSC(I,region_size=size,ratio=0.001)
    superpixelLSC.iterate(10)
    superpixelLSC.enforceLabelConnectivity(min_element_size=25)
    segments = superpixelLSC.getLabels()
    return np.array(segments, np.int64)

def SEEDS_superpixel(I, nseg):
    I = np.array(I[:, :, 0:3], np.float32).copy()
    I_new = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    # I_new =np.array( I[:,:,0:3],np.float32).copy()
    height, width, channels = I_new.shape
    
    superpixelNum = nseg
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, int(superpixelNum), num_levels=2, prior=1,
                                               histogram_bins=5)
    seeds.iterate(I_new, 4)
    segments = seeds.getLabels()
    # segments=SegmentsLabelProcess(segments) # 排除labels中不连续的情况
    return segments

class SubGraphSampler(object):
    def __init__(self, HSI, requires_white=False):
        height, width, bands = HSI.shape  # 原始高光谱数据的三个维度
        if requires_white:
            # 数据standardization标准化,即提前全局BN
            data = np.reshape(HSI, [height * width, bands])
            minMax = preprocessing.StandardScaler()
            data = minMax.fit_transform(data)
            self.data = np.reshape(data, [height, width, bands])
        else: self.data=HSI

    def SegmentsLabelProcess(self,labels):
        '''
        对labels做后处理，防止出现label不连续现象
        '''
        labels = np.array(labels, np.int64)
        H, W = labels.shape
        ls = list(set(np.reshape(labels, [-1]).tolist()))
    
        dic = {}
        for i in range(len(ls)):
            dic[ls[i]] = i
    
        new_labels = labels
        for i in range(H):
            for j in range(W):
                new_labels[i, j] = dic[new_labels[i, j]]
        return new_labels
    
    def ConstructGraph(self,knn_neighbors,n_avgsize=64, compactness=3, max_iter=20, sigma=0.1, min_size_factor=0.5,
                 max_size_factor=3, show_segmap=False):
        '''
        construct graphs
        :param knn_neighbors:
        :param n_avgsize:
        :param compactness:
        :param max_iter:
        :param sigma:
        :param min_size_factor:
        :param max_size_factor:
        :return:
        '''
        # 执行 SLCI 并得到Q(nxm),S(m*b)
        self.knn_neighbors = knn_neighbors
        img = self.data
        (h, w, d) = img.shape
        n_segments = h * w // n_avgsize
        # 计算超像素S以及相关系数矩阵Q
        # segments = slic(img, n_segments=n_segments, compactness=compactness, max_iter=max_iter,
        #                 convert2lab=False, sigma=sigma, enforce_connectivity=True,
        #                 min_size_factor=min_size_factor, max_size_factor=max_size_factor, slic_zero=False)
        
        # segments = felzenszwalb(img, scale=1,sigma=0.5,min_size=25)
    
        # segments = quickshift(img,ratio=1,kernel_size=5,max_dist=4,sigma=0.8, convert2lab=False)
    
        segments=LSC_superpixel(img,n_segments)
    
        # segments=SEEDS_superpixel(img,n_segments)
    
        # 判断超像素label是否连续,否则予以校正
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = self.SegmentsLabelProcess(segments)
        # self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)
    
        # # ######################################显示超像素图片
        if show_segmap:
            out = mark_boundaries(img[:, :, [d//2, d//4, d//2+d//4]], segments)
            plt.figure()
            plt.imshow(out)
            plt.show()
    
        self.segments=segments = np.reshape(segments, [-1])
        self.S=S = np.zeros([superpixel_count, d], dtype=np.float32)
        # Q = np.zeros([w * h, superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])
        
        # positions
        x_coordinate = np.arange(0, h)
        y_coordinate = np.arange(0, w)
        x_coordinate, y_coordinate = np.meshgrid(x_coordinate, y_coordinate)
        xy_coordinate = np.transpose(np.stack([x_coordinate, y_coordinate], axis=-1), (1, 0, 2)).reshape([h * w, 2])  # repaired
        self.positions=positions = np.zeros([superpixel_count, 2], dtype=np.float)
        
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            # Q[idx, i] = 1
            positions[i] = np.sum(xy_coordinate[idx], 0) / count # 计算positions
        
        
        if knn_neighbors>0:
            # 对positions做KNN 建图
            print("constructing graphs...")
            self.edges = knn_graph(torch.from_numpy(positions), knn_neighbors, batch=None, loop=False,
                              flow="target_to_source")
            
            # 方案二 根据光谱相似性 做KNN 建图
            # self.edges = knn_graph(torch.from_numpy(S), knn_neighbors, batch=None, loop=False,
            #                        flow="target_to_source")
            
        else:
            # 第二种建图方案,只建立接壤像素的边
            A = np.zeros([superpixel_count, superpixel_count], dtype=np.float32)
            segments = np.reshape(self.segments, [h, w])
            for i in range(h - 1):
                for j in range(w - 1):
                    sub = segments[i:i + 2, j:j + 2]
                    sub_max = np.max(sub).astype(np.int32)
                    sub_min = np.min(sub).astype(np.int32)
                    if sub_max != sub_min:
                        idx1 = sub_max
                        idx2 = sub_min
                        if A[idx1, idx2] != 0: continue
                
                        # pix1 = self.S[idx1]
                        # pix2 = self.S[idx2]
                        # diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                        # A[idx1, idx2] = A[idx2, idx1] = diss
                        A[idx1, idx2] = A[idx2, idx1]=1
            A_sparse=dense_to_sparse(torch.from_numpy(A))
            self.edges=A_sparse[0]
            self.weights=A_sparse[1]
        return None
    
    def get_graph_samples(self, gt,num_hops,):
        '''
        generate samples (subgraphs) according to the ground truth
        :param GT:
        :return:
        '''
        if len(gt.shape) == 3:
            h, w, _ = gt.shape
            gtFlag = np.sum(gt, axis=-1, keepdims=False)
        else:
            h, w = gt.shape
            gtFlag = gt
        segments = np.reshape(self.segments, [h, w])
        
        sub_graphs_list= []
        for i in range(h):
            for j in range(w):
                if gtFlag[i, j].any() == 0: continue
                node_idx=segments[i,j] # 像素对应节点位置
                sub_graph,sub_edges,center_idx,_ = k_hop_subgraph(int(node_idx),num_hops,self.edges,
                                                                  relabel_nodes=True,flow='target_to_source')
                real_sub_nodes=[]

                idx_list=sub_graph.numpy().tolist()
                for idx in range(len(idx_list)):
                    if idx!=center_idx.numpy()[0]:
                        real_sub_nodes.append(self.S[idx_list[idx]])
                real_sub_nodes = np.array(real_sub_nodes, dtype=np.float32)
                
                sample=torch.from_numpy(real_sub_nodes)
                sub_graphs_list.append(sample)
                
        return sub_graphs_list # neighbors
    
    def getOneGraphSamplePosition(self,pos,num_hops):
        x=pos[0]
        y=pos[1]
        height, width, bands = self.data.shape
        segments = np.reshape(self.segments, [height, width])

        sub_graphs_list = []

        node_idx = segments[x, y]  # 像素对应节点位置
        sub_graph, sub_edges, center_idx, _ = k_hop_subgraph(int(node_idx), num_hops, self.edges,
                                                             relabel_nodes=True, flow='target_to_source')
        real_sub_nodes_pos = []
        
        idx_list = sub_graph.numpy().tolist()

        real_sub_nodes_pos.append(self.positions[idx_list[center_idx.numpy()[0]]])
        
        for idx in range(len(idx_list)):
            if idx != center_idx.numpy()[0]:
                real_sub_nodes_pos.append(self.positions[idx_list[idx]])
        real_sub_nodes_pos = np.array(real_sub_nodes_pos, dtype=np.float32)

        return real_sub_nodes_pos

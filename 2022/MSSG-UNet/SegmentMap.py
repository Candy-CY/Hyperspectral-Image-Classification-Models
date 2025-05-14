import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import skimage.io as io
from skimage.segmentation import slic,mark_boundaries

class SegmentMap(object):
    def __init__(self,datasetname:np.array):
        
        if datasetname=='indian_':
            segs = sio.loadmat('Superpixel_segments\\segmentmaps'+'indian.mat')
            self.segs = segs['segmentmaps']
        if datasetname=='paviaU_':
            segs = sio.loadmat('Superpixel_segments\\segmentmaps'+'paviau.mat')
            self.segs = segs['segmentmaps']
        if datasetname == 'salinas_':
            segs = sio.loadmat('Superpixel_segments\\segmentmaps' + 'salinas.mat')
            self.segs = segs['segmentmaps']
        if datasetname == 'KSC_':
            segs = sio.loadmat('Superpixel_segments\\segmentmaps' + 'ksc.mat')
            self.segs = segs['segmentmaps']
        
    def getHierarchy(self):
        '''
        generate hierarchical node representations
        :param hierarchy: 从大到小的序列,表示每次池化的节点个数
        :return:
        '''
        segs=self.segs
        layers, h, w = self.segs.shape
        segs=np.concatenate([np.reshape( [i for i in range(h*w)],[1,h,w] ), segs],axis=0)
        layers=layers+1
        
        S_list=[]

        # originalNodesIdx =np.reshape(  [i for i in range(h*w)],[h,w] )  # 原始像素节点索引
        
        for i in range(layers-1):
            S=np.zeros([np.max(segs[i])+1,np.max(segs[i+1])+1])
            l1=np.reshape(segs[i],[-1])
            l2=np.reshape(segs[i+1],[-1])
            for x in range(h*w):
                if S[ l1[x] ,l2[x]]!=1: S[ l1[x] ,l2[x]]=1
            S_list.append(S)
        
        # for i in range(len(S_list)):
        #     s=S_list[0]
        #     print(sum(abs(np.sum(s,1)-np.ones(s.shape[0]))  ))
        
        A_list = []
        # '''
        # generating adjacency matrixes for each hierarchy
        # '''
        # superpixelLabels = []
        # for i in range(len(S_list)):
        #     ## 对于每个S,首先映射回原始尺寸
        #     x = [x for x in range(len(S_list[i][0]))]  # 产生x=[]0,1,2,...,Ni]
        #     x = np.array(x, dtype=int)
        #     for j in range(i + 1):
        #         x = np.matmul(S_list[i - j], x)
        #     superpixelLabels.append(x)
        superpixelLabels=self.segs
        # 显示分割图像
        # for i in range(len(superpixelLabels)):
        #     ## 根据映射到原始
        #     map = np.reshape(superpixelLabels[i], [h, w])
        #     out = mark_boundaries(np.zeros_like(self.segs[0]), np.array(map, np.int))
        #     plt.figure()
        #     plt.imshow(out)
        #     plt.show()
    
        # 根据标签计算邻接矩阵
        '''
         根据 segments 判定邻接矩阵
        :return:
        '''
        for l in range(len(superpixelLabels)):
            segments = np.reshape(superpixelLabels[l], [h, w])
            superpixel_count = int(np.max(superpixelLabels[l])) + 1
            A = np.zeros([superpixel_count, superpixel_count], dtype=np.float32)
            (h, w) = (h, w)
            # for i in range(h - 2):
            #     for j in range(w - 2):
            #         sub = segments[i:i + 2, j:j + 2]
            #         sub_max = np.max(sub).astype(np.int32)
            #         sub_min = np.min(sub).astype(np.int32)
            #
            #         if sub_max != sub_min:
            #             idx1 = sub_max
            #             idx2 = sub_min
            #             if A[idx1, idx2] != 0: continue
            #             A[idx1, idx2] = A[idx2, idx1] = 1
            for i in range(h - 1):
                for j in range(w - 1):
                    sub = segments[i:i + 2, j:j + 2]
                    sub_max = np.max(sub).astype(np.int32)
                    sub_min = np.min(sub).astype(np.int32)

                    if sub_max != sub_min:
                        idx1 = sub_max
                        idx2 = sub_min
                        if A[idx1, idx2] != 0: continue
                        A[idx1, idx2] = A[idx2, idx1] = 1
            A_list.append(A)
        
        
        # # 方式二,根据
        # # 获取初始4邻接M
        # M = np.zeros([h * w, h*w])
        # for i in range(h - 1):
        #     for j in range(w - 1):
        #         for (x, y) in [(i + 1, j), (i, j + 1)]:
        #             M[i * w + j, x * w + y] = 1
        #             M[x * w + y, i * w + j] = 1
        # I = np.eye(h * w, h * w,)
        # M = M + I
        #
        # # 产生分层M
        # # M = self.getInitM()
        # M_list=[]
        # # M_list.append(M)
        # for i in range(self.Hierarchy_layer_count):
        #     M=torch.mm(self.S_list[i].t(),torch.mm(M,self.S_list[i]))
        #     M=torch.ceil(M * 0.00000001)
        #     M_list.append(M)
        # return M_list
        return S_list, A_list
    
    
def imShow(data):
    plt.figure()
    plt.imshow(data)
    plt.show()
    
def restoreOutputShape(featureMap,hierarchyIdx,S_list):
    '''
    恢复特征图到原始图像空间
    :param featureMap: 该层的特征图
    :param hierarchyIdx: 第几层
    :return:
    '''
    x=featureMap
    for i in range(hierarchyIdx+1):
        x = np.matmul(S_list[hierarchyIdx - i], x)

    return x
    
if __name__=='__main__':
    # SG=SegmentMap('indian_')
    # data = io.imread("indian.bmp")

    SG = SegmentMap('paviaU_')
    data = io.imread("0paviaU_color.bmp")
    
    
    S_list, A_list = SG.getHierarchy()

    # 实际图片
    
    
    # data = io.imread("my.jpg")
    height, width, bands = data.shape  # 原始高光谱数据的三个维度
    # data = np.reshape(data, [height * width, bands])
    # minMax = preprocessing.StandardScaler()
    # data = minMax.fit_transform(data)
    # data = np.reshape(data, [height, width, bands])



    # test
    # S1 = np.array(S_list[0])
    # S2 = np.array(S_list[1])

    # norm_row_S1 = S1 / (np.sum(S1, 1, keepdims=True))  # 行归一化Q
    S1 = np.array(S_list[0])
    norm_row_S1 = S1  # 行归一化Q
    norm_col_S1 = S1 / (np.sum(S1, 0, keepdims=True))  # 列归一化Q
    img = np.matmul(norm_col_S1.T, np.reshape(data, [-1, bands]))
    img = np.matmul(norm_row_S1, img)
    img = np.reshape(np.array(img, np.uint8), [height, width, -1])[:, :, 0:3]
    imShow(img)

    # norm_row_S2 = S2 / (np.sum(S2, 1, keepdims=True))  # 行归一化Q
    S2 = np.array(S_list[1])
    norm_row_S2 = S2  # 行归一化Q
    norm_col_S2 = S2 / (np.sum(S2, 0, keepdims=True))  # 列归一化Q
    img = np.matmul(norm_col_S2.T, np.matmul(norm_col_S1.T, np.reshape(data, [-1, bands])))
    img = np.matmul(norm_row_S1, np.matmul(norm_row_S2, img))
    img = np.reshape(np.array(img, np.uint8), [height, width, -1])[:, :, 0:3]
    imShow(img)



    S3 = np.array(S_list[2])
    norm_row_S3 = S3  # 行归一化Q
    norm_col_S3 = S3 / (np.sum(S3, 0, keepdims=True))  # 列归一化Q
    img = np.matmul(norm_col_S3.T, np.matmul(norm_col_S2.T, np.matmul(norm_col_S1.T, np.reshape(data, [-1, bands]))) )
    img = np.matmul(norm_row_S1, np.matmul(norm_row_S2, np.matmul(norm_row_S3, img)))
    img = np.reshape(np.array(img, np.uint8), [height, width, -1])[:, :, 0:3]
    imShow(img)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.segmentation import slic,mark_boundaries,felzenszwalb,quickshift,random_walker
from sklearn import preprocessing
import cv2
import math

def LSC_superpixel(I, nseg):
    superpixelNum = nseg
    ratio = 0.075
    size = int(math.sqrt(((I.shape[0] * I.shape[1]) / nseg)))
    superpixelLSC = cv2.ximgproc.createSuperpixelLSC(
        I,
        region_size=size,
        ratio=0.005)
    superpixelLSC.iterate()
    superpixelLSC.enforceLabelConnectivity(min_element_size=25)
    segments = superpixelLSC.getLabels()
    return np.array(segments,np.int64)

def SEEDS_superpixel(I, nseg):
    I=np.array(I[:, :, 0:3], np.float32).copy()
    I_new = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    # I_new =np.array( I[:,:,0:3],np.float32).copy()
    height, width, channels = I_new.shape
    superpixelNum = nseg
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, int(superpixelNum), num_levels=2,prior=1,histogram_bins=5)
    seeds.iterate(I_new,4)
    segments = seeds.getLabels()
    # segments=SegmentsLabelProcess(segments) # 排除labels中不连续的情况
    return segments

def SegmentsLabelProcess(labels):
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


class SLIC(object):
    def __init__(self, HSI,labels, n_segments=1000, compactness=20, max_iter=20, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        # 数据standardization标准化,即提前全局BN
        height, width, bands = HSI.shape  # 原始高光谱数据的三个维度
        data = np.reshape(HSI, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        self.labels=labels
        
    
    def get_Q_and_S_and_Segments(self):
        # 执行 SLCI 并得到Q(nxm),S(m*b)
        img = self.data
        (h, w, d) = img.shape
        # 计算超像素S以及相关系数矩阵Q
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_iter=self.max_iter,
                        convert2lab=False,sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor,slic_zero=False)

        # segments = felzenszwalb(img, scale=1,sigma=0.5,min_size=25)
        
        # segments = quickshift(img,ratio=1,kernel_size=5,max_dist=4,sigma=0.8, convert2lab=False)
        
        # segments=LSC_superpixel(img,self.n_segments)
        
        # segments=SEEDS_superpixel(img,self.n_segments)
        
        # 判断超像素label是否连续,否则予以校正
        if segments.max()+1!=len(list(set(np.reshape(segments,[-1]).tolist()))): segments = SegmentsLabelProcess(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)
        
        # ######################################显示超像素图片
        out = mark_boundaries(img[:,:,[0,1,2]], segments)
        # out = (img[:, :, [0, 1, 2]]-np.min(img[:, :, [0, 1, 2]]))/(np.max(img[:, :, [0, 1, 2]])-np.min(img[:, :, [0, 1, 2]]))
        '''plt.figure()
        plt.imshow(out)
        plt.show()'''
        
        segments = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])
        
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1
        
        self.S = S
        self.Q = Q
        
        return Q, S , self.segments
    
    def get_A(self, sigma: float):
        '''
         根据 segments 判定邻接矩阵
        :return:
        '''
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # if len(sub_set)>1:
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue
                    
                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A[idx1, idx2] = A[idx2, idx1] = diss
        
        return A

class LDA_SLIC(object):
    def __init__(self,data,labels,n_component):
        self.data=data
        self.init_labels=labels
        self.curr_data=data
        self.n_component=n_component
        self.height,self.width,self.bands=data.shape
        self.x_flatt=np.reshape(data,[self.width*self.height,self.bands])
        self.y_flatt=np.reshape(labels,[self.height*self.width])
        self.labes=labels
        
    def LDA_Process(self,curr_labels):
        '''
        :param curr_labels: height * width
        :return:
        '''
        curr_labels=np.reshape(curr_labels,[-1])
        idx=np.where(curr_labels!=0)[0]
        x=self.x_flatt[idx]
        y=curr_labels[idx]
        lda = LinearDiscriminantAnalysis()#n_components=self.n_component
        lda.fit(x,y-1)
        X_new = lda.transform(self.x_flatt)
        return np.reshape(X_new,[self.height, self.width,-1])
       
    def SLIC_Process(self,img,scale=25):
        n_segments_init=self.height*self.width/scale
        print("n_segments_init",n_segments_init)
        myslic=SLIC(img,n_segments=n_segments_init,labels=self.labes, compactness=1,sigma=1, min_size_factor=0.1, max_size_factor=2)
        Q, S, Segments = myslic.get_Q_and_S_and_Segments()
        A=myslic.get_A(sigma=10)
        return Q,S,A,Segments
        
    def simple_superpixel(self,scale):
        curr_labels = self.init_labels
        X = self.LDA_Process(curr_labels)
        Q, S, A, Seg = self.SLIC_Process(X,scale=scale)
        return Q, S, A,Seg
    
    def simple_superpixel_no_LDA(self,scale):
        Q, S, A, Seg = self.SLIC_Process(self.data,scale=scale)
        return Q, S, A,Seg
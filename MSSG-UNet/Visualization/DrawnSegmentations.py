import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import skimage.io as io
from skimage.segmentation import slic, mark_boundaries
import cv2

class SegmentMap(object):
    def __init__(self, datasetname: np.array):
        
        if datasetname == 'indian_':
            segs = sio.loadmat('segments\\segmentmaps' + 'indian.mat')
            self.segs = segs['segmentmaps']
        if datasetname == 'paviaU_':
            segs = sio.loadmat('segments\\segmentmaps' + 'paviau.mat')
            self.segs = segs['segmentmaps']
        if datasetname == 'salinas_':
            segs = sio.loadmat('segments\\segmentmaps' + 'salinas.mat')
            self.segs = segs['segmentmaps']
        if datasetname == 'KSC_':
            segs = sio.loadmat('segments\\segmentmaps' + 'ksc.mat')
            self.segs = segs['segmentmaps']
    
    def show(self):
        '''
        generate hierarchical node representations
        :param hierarchy: 从大到小的序列,表示每次池化的节点个数
        :return:
        '''
        segments = self.segs
        layers, h, w = self.segs.shape
        # segs = np.concatenate([np.reshape([i for i in range(h * w)], [1, h, w]), segs], axis=0)
        # layers = layers + 1

        img_cv = cv2.imread('indian.bmp')  # 读取数据
        img_cv=cv2.resize(img_cv,(145,145))
        img=np.array(img_cv)
        
        for i in range(len(segments)):
            seg=segments[i]
            out = mark_boundaries(img, seg)#,color=(0,0,0)
            # out = mark_boundaries(np.ones_like(img)*255, seg,color=(0,0,0))#
            plt.figure()
            plt.imshow(out)
            plt.xticks([])
            plt.yticks([])
            # plt.show()
            plt.savefig('indian_'+str(i)+'.jpg')
        return 0


def imShow(data):
    plt.figure()
    plt.imshow(data)
    plt.show()


def restoreOutputShape(featureMap, hierarchyIdx, S_list):
    '''
    恢复特征图到原始图像空间
    :param featureMap: 该层的特征图
    :param hierarchyIdx: 第几层
    :return:
    '''
    x = featureMap
    for i in range(hierarchyIdx + 1):
        x = np.matmul(S_list[hierarchyIdx - i], x)
    
    return x


if __name__ == '__main__':
    SG=SegmentMap('indian_')
    # data = io.imread("indian.bmp")
    
    # SG = SegmentMap('paviaU_')
    SG.show()
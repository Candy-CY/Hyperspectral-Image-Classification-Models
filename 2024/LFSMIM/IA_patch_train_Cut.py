import numpy as np
import time
import os
import PIL.Image as Image
from scipy.io import loadmat
from scipy.io import savemat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,recall_score,cohen_kappa_score,accuracy_score
from sklearn.decomposition import PCA

color_mat = loadmat('./data/AVIRIS_colormap.mat')


def sample_gt(gt, train_rate):
    """ generate training gt for training dataset
    Args:
        gt (ndarray): full classmap
        train_rate (float): ratio of training dataset
    Returns:
        train_gt(ndarray): classmap of training data
        test_gt(ndarray): classmap of test data
    """
    indices = np.nonzero(gt)  ##([x1,x2,...],[y1,y2,...])
    X = list(zip(*indices))  ## X=[(x1,y1),(x2,y2),...] location of pixels
    y = gt[indices].ravel()
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_rate > 1:
       train_rate = int(train_rate)
    train_indices, test_indices = train_test_split(X, train_size=train_rate, stratify=y, random_state=0)
    train_indices = [t for t in zip(*train_indices)]   ##[[x1,x2,...],[y1,y2,...]]
    test_indices = [t for t in zip(*test_indices)]
    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    
    return train_gt, test_gt


def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]
    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi



patch_size=11
numComponents = 200
data = loadmat('./data/IndianPine.mat')
input = data['input'] #(145,145,200)

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

input = applyPCA(input,numComponents)


TR = data['TR']
TE = data['TE']
LABEL = TR + TE

train_gt, test_gt = sample_gt(LABEL,train_rate=0.05)
LABEL1 = train_gt


# normalize data by band norm
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
# # # data size

height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))
X = mirror_hsi(height,width,band,input_normalize,patch=patch_size)

p = patch_size // 2
gt = LABEL1[:]
LABEL1 = np.pad(LABEL1,(p,p),'constant',constant_values = 0)
x_pos, y_pos = np.nonzero(gt)

indices = np.array([(x,y) for x,y in zip(x_pos, y_pos)])
for i in range(len(indices)):
    x, y = indices[i]
    tg_x = x + p
    tg_y = y + p
    x1, y1 = tg_x - patch_size // 2, tg_y - patch_size // 2
    x2, y2 = x1 + patch_size, y1 + patch_size
    data = X[x1:x2, y1:y2, :]
    label = LABEL1[tg_x, tg_y]
    data = np.asarray(data, dtype='float16')
    label = np.asarray(label, dtype='int8')
    np.savez('data/dataset/IA/train/'+str(i)+'.npz',img=data,label=label)#将切割得到的小图片存在path_out路径下


LABEL2 = test_gt
p = patch_size // 2
gt = LABEL2[:]
LABEL2 = np.pad(LABEL2,(p,p),'constant',constant_values = 0)
x_pos, y_pos = np.nonzero(gt)
indices = np.array([(x,y) for x,y in zip(x_pos, y_pos)])
for i in range(len(indices)):
    x, y = indices[i]
    tg_x = x + p
    tg_y = y + p
    x1, y1 = tg_x - patch_size // 2, tg_y - patch_size // 2
    x2, y2 = x1 + patch_size, y1 + patch_size
    data = X[x1:x2, y1:y2]
    label = LABEL2[tg_x, tg_y]
    data = np.asarray(data, dtype='float16')
    label = np.asarray(label, dtype='uint8')
    np.savez('data/dataset/IA/test/'+str(i)+'.npz',img=data,label=label)#将切割得到的小图片存在path_out路径下
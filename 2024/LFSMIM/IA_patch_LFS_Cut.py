import numpy as np
import time
import os
import PIL.Image as Image
from scipy.io import loadmat
from scipy.io import savemat
from sklearn.decomposition import PCA
import torch

color_mat = loadmat('./data/AVIRIS_colormap.mat')


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
numComponents = 30
data = loadmat('./data/IndianPine.mat')
input = data['input'] #(145,145,200)
TR = data['TR']
TE = data['TE']
LABEL = TR + TE

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX
input = applyPCA(input,numComponents)

# normalize data by band norm
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
# data size
height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))


X = mirror_hsi(height,width,band,input_normalize,patch=patch_size)

mean = np.zeros((1,input.shape[-1]))
std = np.zeros((1,input.shape[-1]))

for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    # input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
    mean[:,i] = X[:,:,i].mean()
    std[:,i] = X[:,:,i].std()

H,W,C = X.shape

imgs = X.transpose(2,0,1)
imgs = torch.tensor(imgs)

rad = 39
mean = mean.reshape(numComponents,1,1)
std = std.reshape(numComponents,1,1)
imgs = imgs * std + mean
freq_imgs = torch.fft.fft2(imgs)

freq_imgs = torch.fft.fftshift(freq_imgs, dim=(-2, -1))
c, h, w = imgs.shape
low_pass_filter = torch.ones((numComponents, h, w))
for i in range(h):
    for j in range(w):
        if (i - (h - 1) / 2)**2 + (j -
                                    (w - 1) / 2)**2 > rad**2:
            low_pass_filter[:, i, j] = 0
# low pass images
low_pass_imgs = freq_imgs * low_pass_filter
low_pass_imgs = torch.fft.ifft2(low_pass_imgs)
low_pass_imgs = torch.abs(low_pass_imgs)

low_pass_imgs = (low_pass_imgs - mean) / std
low_pass_imgs = low_pass_imgs.reshape(H,W,numComponents)



p = patch_size // 2
gt = LABEL[:]
LABEL = np.pad(LABEL,(p,p),'constant',constant_values = 0)
x_pos, y_pos = np.nonzero(gt)
indices = np.array([(x,y) for x,y in zip(x_pos, y_pos)])

for i in range(len(indices)):
    x, y = indices[i]
    tg_x = x + p
    tg_y = y + p
    x1, y1 = tg_x - patch_size // 2, tg_y - patch_size // 2
    x2, y2 = x1 + patch_size, y1 + patch_size
    data = X[x1:x2, y1:y2,:]
    resume = low_pass_imgs[x1:x2, y1:y2,:]
    label = LABEL[tg_x, tg_y]
    data = np.asarray(data, dtype='float16')
    label = np.asarray(label, dtype='int8')
    np.savez('data/dataset/IndianPines/'+str(i)+'.npz',img=data,resume=resume,label=label)#将切割得到的小图片存在path_out路径下


import os, sys
import numpy as np
from torchvision import transforms 
import matplotlib.pyplot as plt


def show_spectral_curve(image, X, Y, total = 3):
    if X is None or Y is None:
        return
    if type(image) != np.ndarray:
        image = image.numpy()
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    if len(image.shape) == 5:
        image = image[0, 0, :, :, :]

    # image shape (spectral, w, h)
    X = X[0]
    Y = Y[0]
    num = 0
    w,h = Y.shape
    for i in range(w):
        for j in range(h):
            if num > total:
                break 
            if Y[i,j] > 0:
                ss = list(image[:, i, j])
                real_ss = list(X[:, i, j])
                ii = list(range(len(ss)))
                plt.plot(ii, ss, label='pred')
                plt.plot(ii, real_ss, label='real')
                num += 1

def show_tensor_image(image, rgb = (0, 1, 2)):
    if type(image) != np.ndarray:
        image = image.numpy()
    r,g,b = rgb
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    if len(image.shape) == 5:
        image = image[0, 0, :, :, :]
    if image.shape[-1] > 3:
        rimg = image[r, :, :]
        gimg = image[g, :, :]
        bimg = image[b, :, :]
    image = np.stack([rimg, gimg, bimg])
    
    def trans(x):
        if type(x) == np.ndarray:
            return np.transpose(x, (1,2,0))
        else:
            return x.permute(1,2,0)

    def totype(x):
        if type(x) == np.ndarray:
            return x.astype(np.uint8)
        else:
            return x.numpy().astype(np.uint8)
    
    reverse_transforms = transforms.Compose([
        # transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: trans(t)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: totype(t)),
        transforms.ToPILImage(),
    ])
        # Take first image of batch
    plt.imshow(reverse_transforms(image))



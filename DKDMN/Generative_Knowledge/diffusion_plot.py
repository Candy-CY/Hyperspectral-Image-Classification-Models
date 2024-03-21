import os,sys
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import torchvision
from torchvision import transforms 
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt

# from new_data import HSIDataLoader, TestDS, TrainDS  # original
from data import HSIDataLoader, TestDS, TrainDS
from unet3d import SimpleUnet
# from spectral_transformer import SpectralTransNet
# from unet import UNetModel
from diffusion import Diffusion
from utils import AvgrageMeter, recorder, show_img

batch_size = 128
patch_size = 16
select_spectral = []
spe = 144
channel = 1 # 3d channel

epochs = 100000 # Try more!
lr = 1e-4
T = 500

rgb = [0, 99, 199]
model_load_path = "./data/save_model/unet3d_patch16_without_downsample_kernal5_fix"
model_name = "unet3d_27000.pkl"

TList = [5, 10, 100, 200, 400]

device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_by_imgs(imgs, rgb = [1, 100, 199]):
    assert len(imgs) > 0
    batch, c, s, h, w = imgs[0].shape
    for i in range(batch):
        plt.figure(figsize = (12, 8))
        for j in range(len(imgs)):
            plt.subplot(1, len(imgs), j + 1)
            img = imgs[j][i, 0, rgb, :, :]
            show_img(img)
        plt.show()            
    
def plot_by_images_v2(imgs, rgb = [1, 100, 199]):
    '''
    input image shape is (spectral, height, width)
    '''
    assert len(imgs) > 0
    s, h, w = imgs[0].shape
    plt.figure(figsize = (12, 8))
    for j in range(len(imgs)):
        plt.subplot(1, len(imgs), j + 1)
        img = imgs[j][rgb, :, :]
        show_img(img)
    plt.show()            
    
def plot_spectral(x0, recon_x0, num = 3):
    '''
    x0, recon_x0 shape is (batch, channel, spectral, h, w)
    '''
    batch, c, s, h, w = x0.shape
    step = h // num
    plt.figure(figsize = (20, 5))
    for ii in range(num):
        i = ii * step 
        x0_spectral = x0[0, 0, :, i, i]
        recon_x0_spectral = recon_x0[0, 0, :, i, i]
        plt.subplot(1, num, ii + 1)
        plt.plot(x0_spectral, label = "x0")
        plt.plot(recon_x0_spectral, label = "recon")
        plt.legend()
    plt.show()
    
def recon_all_fig(diffusion, model, splitX, dataloader, big_img_size = [145, 145]):
    '''
    X shape is (spectral, h, w) => (batch, channel=1, 200, 145, 145)
    '''
    # 1. reconstruct
    t = torch.full((1, ), diffusion.T-1, device = device, dtype = torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(torch.from_numpy(splitX.astype('float32')), t, device)
    _, recon_from_xt = diffusion.reconstruct(model, xt = xt, tempT = t, num = 5)
    
    # ---just for test---
    # recon_from_xt.append(torch.from_numpy(splitX.astype('float32')))
    # plot_by_imgs(recon_from_xt, rgb=rgb)
    # ---------

    res_xt_list = []
    for tempxt in recon_from_xt:
        big_xt = dataloader.split_to_big_image(tempxt.numpy()) 
        res_xt_list.append(big_xt)
    ori_data, _ = dataloader.get_ori_data()
    res_xt_list.append(ori_data)
    plot_by_images_v2(res_xt_list, rgb = rgb)
    
def sample_by_t(diffusion, model, X):
    num = 10
    choose_index = [3]
    x0 = torch.from_numpy(X[choose_index, :, :, :, :]).float()

    step = diffusion.T // num
    for ti in range(10, diffusion.T, step):
        t = torch.full((1, ), ti, device = device, dtype = torch.long)
        xt, tmp_noise = diffusion.forward_diffusion_sample(x0, t, device)
        _, recon_from_xt = diffusion.reconstruct(model, xt = xt, tempT = t, num = 5)
        recon_x0 = recon_from_xt[-1]
        recon_from_xt.append(x0)
        print('---', ti, '---')
        plot_by_imgs(recon_from_xt, rgb = rgb)
        print("x0", x0.shape, "recon_x0", recon_x0.shape)
        plot_spectral(x0, recon_x0)

def inference_by_t(dataloader, diffusion, model, X, ti):
    '''
    X shape is (batch, channel, spe, h, w)
    '''
    X = torch.from_numpy(X).float()
    t = torch.full((1, ), ti, device = device, dtype = torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(X, t, device)

    # 2. 对模型在该t下进行完全恢复尝试验证
    choose_index = [3]
    show_x0 = X[choose_index, :, :, :, :]
    show_xt = xt[choose_index, :, :, :, :]
    _, recon_from_xt = diffusion.reconstruct(model, xt = show_xt, tempT = t, num = 5) # recon_from_xt[0] shape (batch, channel, spe, h, w)
    recon_x0 = recon_from_xt[-1]
    recon_from_xt.append(show_x0)
    print('---', ti, '---')
    plot_by_imgs(recon_from_xt, rgb = rgb)
    plot_spectral(show_x0, recon_x0)

def sample_eval(diffusion, model, X):
    all_size, channel, spe, h, w = X.shape
    num = 16
    step = all_size // num
    r,g,b = 1, 100, 199
    choose_index = list(range(0, all_size, step))
    x0 = torch.from_numpy(X[choose_index, :, :, :, :]).float()

    use_t = 499
    # from xt
    t = torch.full((1, ), use_t, device = device, dtype = torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(x0, t, device)
    _, recon_from_xt = diffusion.reconstruct(model, xt = xt, tempT = t, num = 10)
    recon_from_xt.append(x0)
    plot_by_imgs(recon_from_xt, rgb=rgb)
    
    # from noise
    t = torch.full((1, ), use_t, device = device, dtype = torch.long)
  
    _, recon_from_noise = diffusion.reconstruct(model, xt = x0, tempT = t, num = 10, from_noise = True, shape = x0.shape)
    plot_by_imgs(recon_from_noise, rgb = rgb)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("save model done. path=%s" % path)

def plot():
    dataloader = HSIDataLoader({"data":{"data_sign":"Indian", "padding":False, "batch_size":batch_size, "patch_size":patch_size, "select_spectral":select_spectral}})
    train_loader, X, Y = dataloader.generate_torch_dataset(light_split=True)
    diffusion = Diffusion(T = T)

    model = SimpleUnet(_image_channels=channel)

    model_path = "%s/%s" % (model_load_path, model_name)
    model.load_state_dict(torch.load(model_path))
  
    model.to(device)
    
    # for ti in TList:
    #     inference_by_t(dataloader, diffusion, model, X, ti)
    #     print("feature extract t=%s done." % ti)
    
    sample_eval(diffusion, model, X)
    print('done.')

if __name__ == "__main__":
    plot()

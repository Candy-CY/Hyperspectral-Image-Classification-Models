import torch
import torchvision
from torchvision import transforms 
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os, sys
from data import HSIDataLoader, TestDS, TrainDS
from unet3d import SimpleUnet
from diffusion import Diffusion
from utils import AvgrageMeter, recorder, show_img
from utils import device


# for PU
# sign = 'PU'
# batch_size = 20
# # patch_size = 64 # original
# patch_size = 16
# select_spectral = []
# spe = 104
# channel = 1 # 3d channel

# for IP
# sign = 'IP'
# batch_size = 20
# patch_size = 64
# select_spectral = []
# spe = 200
# channel = 1 # 3d channel

# for SA
sign = 'SA'
batch_size = 8
patch_size = 16
select_spectral = []
spe = 208
channel = 1 # 3d channel

# for KSC
# sign = 'KSC'
# batch_size = 20
# patch_size = 16
# select_spectral = []
# spe = 176
# channel = 1 # 3d channel


# epochs = 100000 # more than 30000
epochs = 10000
lr = 1e-4
T = 500

rgb = [30, 50, 90]
path_prefix = "./save_model/%s_diffusion" % sign

# device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_by_imgs(imgs, rgb=[1, 100, 199]):
    assert len(imgs) > 0
    batch, c, s, h, w = imgs[0].shape
    for i in range(batch):
        plt.figure(figsize = (12, 8))
        for j in range(len(imgs)):
            plt.subplot(1, len(imgs), j + 1)
            img = imgs[j][i, 0, rgb, :, :]
            show_img(img)
        plt.show()            
    
def plot_by_images_v2(imgs, rgb=[1, 100, 199]):
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
    
def plot_spectral(x0, recon_x0, num=3):
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
    t = torch.full((1, ), diffusion.T-1, device=device, dtype=torch.long)
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
    plot_by_images_v2(res_xt_list, rgb=rgb)
    
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
        print('---',ti,'---')
        plot_by_imgs(recon_from_xt, rgb=rgb)
        plot_spectral(x0, recon_x0)

def sample_eval(diffusion, model, X):
    all_size, channel, spe, h, w = X.shape
    num = 5
    step = all_size // num
    r, g, b = 1, 100, 199
    choose_index = list(range(0, all_size, step))
    x0 = torch.from_numpy(X[choose_index, :, :, :, :]).float()

    use_t = 499
    # from xt
    t = torch.full((1, ), use_t, device = device, dtype = torch.long)
    xt, tmp_noise = diffusion.forward_diffusion_sample(x0, t, device)
    _, recon_from_xt = diffusion.reconstruct(model, xt = xt, tempT = t, num = 10)
    recon_from_xt.append(x0)
    plot_by_imgs(recon_from_xt, rgb = rgb)
    
    # from noise
    t = torch.full((1, ), use_t, device = device, dtype = torch.long)
    _, recon_from_noise = diffusion.reconstruct(model, xt = x0, tempT = t, num = 10, from_noise = True, shape = x0.shape)
    plot_by_imgs(recon_from_noise, rgb = rgb)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("save model done. path=%s" % path)

def train():
    dataloader = HSIDataLoader(
        {"data":{"data_sign": "Salinas", "padding": False, "batch_size": batch_size, "patch_size": patch_size, "select_spectral": select_spectral}})
    train_loader, X, Y = dataloader.generate_torch_dataset(light_split = True)
    diffusion = Diffusion(T = T)
    model = SimpleUnet(_image_channels = channel)
    # model = DinD(inp_channel = spe)
    # model = UNet2D(img_channels = spe)
    model.to(device)
    optimizer = Adam(model.parameters(), lr = lr)

    loss_metric = AvgrageMeter()

    # assert not os.path.exists(path_prefix)
    # os.makedirs(path_prefix)
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)

    for epoch in range(1, epochs + 1):
        loss_metric.reset()
        for step, (batch, _) in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            cur_batch_size = batch.shape[0]
            t = torch.randint(0, diffusion.T , (cur_batch_size,), device = device).long()
            loss, temp_xt, temp_noise, temp_noise_pred = diffusion.get_loss(model, batch, t)
            loss.backward()
            optimizer.step()
            loss_metric.update(loss.item(), batch.shape[0])

            if step % 10 == 0:
                print(f"[Epoch-step] {epoch} | step {step:03d} Loss: {loss.item()} ")
        print("[TRAIN EPOCH %s] loss=%s" % (epoch, loss_metric.get_avg()))

        if epoch % 1000 == 0:
            #sample_by_t(diffusion, model, X)
            #sample_eval(diffusion, model, X)
            _, splitX, splitY = dataloader.generate_torch_dataset(split = True)
            # recon_all_fig(diffusion, model, splitX, dataloader, big_img_size = [145, 145])
            path = "%s/unet3d_%s.pkl" % (path_prefix, epoch)
            save_model(model, path)


if __name__ == "__main__":
    train()

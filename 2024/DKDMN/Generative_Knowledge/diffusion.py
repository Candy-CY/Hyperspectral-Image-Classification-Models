import torch
import torchvision
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F
from data import HSIDataLoader
import numpy as np
from plot import show_tensor_image
from utils import device


class Diffusion(object):
    def __init__(self, T = 1000) -> None:
        self.T = T
        self.betas = self._linear_beta_schedule(timesteps = self.T)
        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[: -1], (1, 0), value = 1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    def _linear_beta_schedule(self, timesteps, start = 0.0001, end = 0.02):
        return torch.linspace(start, end, timesteps)

    def _get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


    def get_loss(self, model, x_0, t):
        x_noisy, noise = self.forward_diffusion_sample(x_0, t, device)
        noise_pred = model(x_noisy, t)
        if x_noisy.shape[1] == 1:
            x_noisy = np.squeeze(x_noisy, axis=1)
            noise = np.squeeze(noise, axis=1)
        else:
            pass
        return F.l1_loss(noise, noise_pred), x_noisy, noise, noise_pred


    @torch.no_grad()
    def sample_timestep(self, x, t, model):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        x is xt, t is timestamp
        return x_{t-1}
        """
        betas_t = self._get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self._get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    @torch.no_grad()
    def reconstruct(self, model, xt=None, tempT=None, num = 5, from_noise=False, shape=None):
        '''
        分别从纯noise和xt, 逐步恢复信息
        如果不给定xt 则自动使用随机造成
        给定xt同时需要给定tempT, 表明该xt是来自多少步造成生成
        '''
        stepsize = int(tempT.cpu().numpy()[0] / num)
        index = []
        res = []
        # Sample noise
        if from_noise:
            img = torch.randn(shape, device = device)
        else:
            img = xt

        if tempT is None:
            tempT = self.T

        for i in range(0, tempT)[::-1]:
            t = torch.full((1, ), i, device = device, dtype = torch.long)
            img = self.sample_timestep(img, t, model)
            if i % stepsize == 0:
                index.append(i)
                res.append(img.detach().cpu())
        index.append(i)
        res.append(img.detach().cpu())
        return index, res

    @torch.no_grad()
    def reconstruct_v2(self, model, xt = None, tempT = None, use_index = [], from_noise = False, shape = None):
        '''
        分别从纯noise和xt, 逐步恢复信息
        如果不给定xt 则自动使用随机造成
        给定xt同时需要给定tempT, 表明该xt是来自多少步造成生成
        '''
        index = []
        res = []
        # Sample noise
        if from_noise:
            img = torch.randn(shape, device=device)
        else:
            img = xt

        if tempT is None:
            tempT = self.T

        for i in range(0, tempT)[::-1]:
            t = torch.full((1, ), i, device = device, dtype = torch.long)
            img = self.sample_timestep(img, t, model)
            if i in use_index:
                index.append(i)
                res.append(img.detach().cpu())
        index.append(i)
        res.append(img.detach().cpu())
        return index, res

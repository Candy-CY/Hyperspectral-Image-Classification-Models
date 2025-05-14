import os
import random
from tqdm import tqdm
import numpy as np
from scipy import ndimage

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.optim import AdamW
from timm.scheduler import CosineLRScheduler

from Models import HSIMAE
from Utils.Preprocessing import get_data_cut_file
from Utils.Seed_Everything import seed_everything, stable

import warnings
warnings.filterwarnings('ignore')


class HSIdataset4PT(data.Dataset):
    def __init__(self, data_cubes, train=False, device='cuda:0'):
        self.data_cubes = data_cubes[0]
        self.cut_info = data_cubes[1]
        self.train = train
        self.device = device

    def random_horizontal_filp(self, data, r=0.5):
        if random.random() < r:
            return np.flip(data, 1)
        else:
            return data

    def random_vertical_filp(self, data, r=0.5):
        if random.random() < r:
            return np.flip(data, 0)
        else:
            return data

    def __getitem__(self, index):
        c, h, w, num, max_, min_ = self.cut_info[index]
        data_cube = self.data_cubes[num]
        data = data_cube[h: (h + 9), w: (w + 9), :]
        data = (data - min_) / (max_ - min_)

        if self.train:
            data = self.random_horizontal_filp(data)
            data = self.random_vertical_filp(data)
        data = torch.tensor(data.copy(), dtype=torch.float32)
        data = data.unsqueeze(0).permute(0, 3, 1, 2)
        return data

    def __len__(self):
        return len(self.cut_info)


def mask_pretraining(data_cubes, save_path, model_name, img_size=9, bands=32, mask_ratio=0.50, lr=5e-3, wd=5e-2, bs=512,
                     epochs=100, depth=12, dim=64, s_depth=6, dec_dim=48, dec_depth=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = HSIdataset4PT(data_cubes, train=True)
    del data_cubes

    print('dataset load finished')
    print('训练集大小：' + str(len(train_dataset)))
    print('网络参数：', [dim, depth, dec_dim, dec_depth])

    model = HSIMAE(img_size=img_size, patch_size=3, in_chans=1, bands=bands, b_patch_size=8,
                   embed_dim=dim, depth=depth, num_heads=dim // 16, s_depth=s_depth,
                   decoder_embed_dim=dec_dim, decoder_depth=dec_depth, decoder_num_heads=dec_dim // 8,
                   norm_pix_loss=True, trunc_init=True).to(device)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_dataload = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
    print('batch load finished')
    print('训练轮次：' + str(len(train_dataload)))

    no_decay = ['bias', 'norm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': wd},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    iters = epochs * len(train_dataload)
    scheduler = CosineLRScheduler(optimizer, t_initial=iters, lr_min=1e-6, warmup_t=int(np.ceil(iters * 0.05)))

    epoch_loss_list = []
    val_loss_list = []
    iter_num = 0
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        model.train()
        for x in stable(train_dataload, 42 + epoch):
            inputs = x.to(device)
            loss, outputs, _ = model(inputs, mask_ratio=mask_ratio)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step(iter_num)
            iter_num += 1
            train_loss += loss.item()

        tloss = train_loss / len(train_dataload)
        epoch_loss_list.append(tloss)

    torch.save(model.state_dict(), os.path.join(save_path, model_name))
    history = np.array([epoch_loss_list, val_loss_list])
    np.save(save_path + '/train_log.npy', history)


if __name__ == "__main__":
    seed_everything(42)

    data_paths = ['numpy_files']  # data shape is [h, w, Bands]

    img_size = 9
    data_cubes = get_data_cut_file(data_paths, patch_size=img_size, norm=False, GWPCA=True, ratio=1.0)

    bands = 32
    mask_ratio = 0.5
    lr = 5e-3
    wd = 5e-2
    batch_size = 512
    epoch = 10
    enc_paras = [12, 256, 9]  # [depth, dimension, spatial-spectral encoders depth], [12, 128, 9] for HSIMAE-Base, [12, 256, 9] for Large
    dec_paras = [8, 64]  # [depth, dimension]

    save_path = "save path"
    model_name = 'HSIMAE_L.pkl'

    seed_everything(42)
    mask_pretraining(data_cubes,
                     save_path,
                     model_name,
                     img_size=img_size,
                     bands=bands,
                     bs=batch_size,
                     mask_ratio=mask_ratio,
                     lr=lr,
                     wd=wd,
                     epochs=epoch,
                     depth=enc_paras[0],
                     dim=enc_paras[1],
                     s_depth=enc_paras[2],
                     dec_depth=dec_paras[0],
                     dec_dim=dec_paras[1],
                     )
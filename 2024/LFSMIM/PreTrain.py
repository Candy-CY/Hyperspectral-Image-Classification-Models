import torch
import argparse
import torch.nn as nn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os
import PIL.Image as Image

class RandomMaskingGenerator:
    def __init__(self, number_patches, mask_ratio):
        self.number_patches = number_patches
        self.num_mask = int(mask_ratio * self.number_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.number_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.number_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask 

seed = 2333
pca_num = 30
patch_size = 11

data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = 'data/dataset/IndianPines/'

class MapDatasetLoader(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transform = transforms
        self.files = os.listdir(self.path)
        self.mask_gen = RandomMaskingGenerator(number_patches=pca_num, mask_ratio=0.5)
        self.sp_mask = RandomMaskingGenerator(number_patches=patch_size*patch_size, mask_ratio=0.5)
        # self.pixel_mean = [123.675, 116.28, 103.53]
        # self.pixel_std = [58.395, 57.12, 57.375]
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        filename = self.files[index]
        data = np.load(self.path+filename) 
        # color cy label
        img = data['img'].transpose(2,0,1)
        resume = data['resume'].transpose(2,0,1)
        label = data['label']
        mask = torch.from_numpy(self.mask_gen()).float() > 0
        mask_sp = torch.from_numpy(self.sp_mask()).float() > 0 
        return torch.tensor(img).float(),torch.tensor(resume).float(),torch.tensor(label),mask,mask_sp

dataset = MapDatasetLoader(path, transforms=data_transform)
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)



data = loadmat('./data/IndianPine.mat')
color_mat = loadmat('./data/AVIRIS_colormap.mat')


from MyViT import PretrainVisionTransformer
model = PretrainVisionTransformer(image_size=patch_size,near_band=1,mlp_dim=8,num_patches=pca_num,encoder_dim=64,encoder_dim_head=16,decoder_num_classes=patch_size*patch_size).to(device)

model.train()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400//10, gamma=0.9, verbose=True) 

tic = time.time()
for epoch in range(400):
    epoch_loss = 0
    for batch_data in train_dataloader:
        batch_input = batch_data[0] 
        B, C, s, _ = batch_input.shape
        batch_input = batch_input.reshape(B,C,s*s).to(device)

        batch_resume = batch_data[1] 

        batch_mask = batch_data[3]
        batch_target = batch_resume[batch_mask].reshape(B, -1, s*s).to(device) 

        mask_sp = batch_data[4].to(device) 

        batch_pred = model(batch_input,batch_mask)
        optimizer.zero_grad()
        loss = criterion(batch_pred, batch_target)
        # break
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss.item()
    print('epcoch',str(epoch),'--:',epoch_loss)


def save_model(epoch, model, optimizer):
    output_dir = 'data'
    checkpoint_path = output_dir + '/Low_mask50_PCA_patch11_IA_200v1'
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(to_save, checkpoint_path)    
save_model(model=model, optimizer=optimizer,epoch=epoch)

toc = time.time()
print("Running Time: {:.2f}".format(toc-tic))
print("**************************************************")


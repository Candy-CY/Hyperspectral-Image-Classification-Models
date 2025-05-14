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
from collections import OrderedDict
from utils import *
from sklearn.metrics import confusion_matrix,classification_report,recall_score,cohen_kappa_score,accuracy_score

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float16)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

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
    
seed = 0
pca_num = 200
patch_size = 11

data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = 'data/dataset/IA/train/'

class MapDatasetLoader(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transform = transforms
        self.files = os.listdir(self.path)
        self.mask_gen = RandomMaskingGenerator(number_patches=pca_num, mask_ratio=0.75)
        # self.pixel_mean = [123.675, 116.28, 103.53]
        # self.pixel_std = [58.395, 57.12, 57.375]
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        filename = self.files[index]
        data = np.load(self.path+filename) 
        # color cy label
        img = data['img'].transpose(2,0,1)
        label = data['label'] - 1
        mask = torch.from_numpy(self.mask_gen())
        mask = mask > 0
        return torch.tensor(img).float(),torch.tensor(label),mask

dataset = MapDatasetLoader(path, transforms=data_transform)
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

from MyViT import VisionTransformerEncoder

model = VisionTransformerEncoder(image_size=patch_size,near_band=1,num_patches=pca_num,dim=patch_size*patch_size,dim_head=24,num_classes=16,mlp_dim=12,dropout=0.1,emb_dropout=0.1,mode='CAF')

model_state_dict = model.state_dict()
model_key = list(model_state_dict.keys())
checkpoint = torch.load('data/Low_mask75_patch11_IA_400', map_location='cpu') #low 
print("Load ckpt from %s" % 'data/model_low_weight')
checkpoint_model = None
for model_key in 'model|module'.split('|'):
    if model_key in checkpoint:
        checkpoint_model = checkpoint[model_key]
        print("Load state_dict by model_key = %s" % model_key)
        break

if checkpoint_model is None:
    checkpoint_model = checkpoint
state_dict = model.state_dict()

for k in ['mlp_head.weight', 'mlp_head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

all_keys = list(checkpoint_model.keys())
new_dict = OrderedDict()
for key in all_keys:
    if key.startswith('backbone.'):
        new_dict[key[9:]] = checkpoint_model[key]
    elif key.startswith('encoder.'):   
        new_dict[key[8:]] = checkpoint_model[key]
    elif key.startswith('decoder.'):  
        continue
    else:
        new_dict[key] = checkpoint_model[key]
        # continue
checkpoint_model = new_dict

if 'pos_embedding' in checkpoint_model:
    pos_embed_checkpoint = checkpoint_model['pos_embedding']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.num_patches
    num_extra_tokens = model.pos_embedding.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embedding'] = new_pos_embed

state_dict = model.state_dict()
for model_key in list(state_dict.keys()):
    if model_key in list(checkpoint_model.keys()):
        state_dict[model_key] = checkpoint_model[model_key]
        print("Load state_dict by model_key = %s" % model_key)
    else:
        list(state_dict.keys())
model.load_state_dict(state_dict, strict=False)
# load_state_dict(model, state_dict, prefix='')




model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300//10, gamma=0.9, verbose=True) 

model.train()
tic = time.time()
for epoch in range(300):
    epoch_loss = 0
    for batch_data in train_dataloader:
        img = batch_data[0] 
        B, C, s, _ = img.shape
        img = img.reshape(B,C,s*s).to(device)
        label = batch_data[1].to(device)
        batch_pred = model(img,None)
        optimizer.zero_grad()
        loss = criterion(batch_pred,label.long())
        # break
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss.item()
    
    print('epcoch',str(epoch),'--:',epoch_loss)
 

path = 'data/dataset/IA/test/'
class MaptestDatasetLoader(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transform = transforms
        self.files = os.listdir(self.path)
        self.mask_gen = RandomMaskingGenerator(number_patches=200, mask_ratio=0.75)
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        filename = self.files[index]
        data = np.load(self.path+filename) 
        # color cy label
        img = data['img'].transpose(2,0,1)
        label = data['label']-1
        mask = torch.from_numpy(self.mask_gen())
        mask = mask > 0
        return torch.tensor(img).float(),torch.tensor(label),mask

test_dataset = MaptestDatasetLoader(path, transforms=data_transform)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

model.eval()

pred_map = []
true_map = []
infer_st = time.time() 

for batch_idx, batch_data in enumerate(test_dataloader):
    img = batch_data[0] 
    B, C, s, _ = img.shape
    img = img.reshape(B,C,s*s).to(device)
    label = batch_data[1].to(device)
    batch_pred = model(img,None)
    _, pred = torch.max(batch_pred, dim = 1)
    pred_map.append(np.array(pred.detach().cpu())) 
    true_map.append(np.array(label.detach().cpu()))
infer_time = time.time() - infer_st
print(f"inferring time: {infer_time}")

pred_tr = []
label_tr = []
for i in range(len(pred_map)):
    for j in range(len(pred_map[i])):
        pred_tr.append(pred_map[i][j])
        label_tr.append(true_map[i][j])

pred_map = np.array(pred_tr)
true_map = np.array(label_tr)

OA1, AA_mean1, Kappa1, AA1 = output_metric(true_map, pred_map)
print('OA:',OA1,'AA:',AA1,'AAmean:',AA_mean1,'K:',Kappa1)


file_name = "classification_report.txt"
with open(file_name, 'w') as f:  # 设置文件对象  
    print(AA1[0]*100,file=f)
    print(AA1[1]*100,file=f)
    print(AA1[2]*100,file=f)
    print(AA1[3]*100,file=f)
    print(AA1[4]*100,file=f)
    print(AA1[5]*100,file=f)
    print(AA1[6]*100,file=f)
    print(AA1[7]*100,file=f)
    print(AA1[8]*100,file=f)
    print(AA1[9]*100,file=f)
    print(AA1[10]*100,file=f)
    print(AA1[11]*100,file=f)
    print(AA1[12]*100,file=f)
    print(AA1[13]*100,file=f)
    print(AA1[14]*100,file=f)
    print(AA1[15]*100,file=f)
    
    print(OA1*100,file=f)
    print(AA_mean1*100,file=f)
    print(Kappa1*100,file=f)





# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:33:53 2021

@author: HLB
"""

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class HyperData(Dataset):
    def __init__(self, dataset, transfor):
        self.data = dataset[0].astype(np.float32)
        self.transformer = transfor
        self.labels = []
        for n in dataset[1]:
            self.labels += [int(n)]

    def __getitem__(self, index):
        label = self.labels[index]
        if self.transformer == None:
            img = torch.from_numpy(np.asarray(self.data[index,:,:,:]))
            return img, label
        elif len(self.transformer) == 2:
            img = torch.from_numpy(np.asarray(self.transformer[1](self.transformer[0](self.data[index,:,:,:]))))   
            return img, label
        else:
            img = torch.from_numpy(np.asarray(self.transformer[0](self.data[index,:,:,:])))   
            return img, label            

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels



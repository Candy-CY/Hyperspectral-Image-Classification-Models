import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class HData(Dataset):
	def __init__(self, dataset, tranform=None):
		self.data = dataset[0]
		self.trans=tranform
		self.labels = dataset[1]
		#print("shape: ",self.labels.shape)
		#for n in dataset[1]: self.labels += [int(n)]

	def __getitem__(self, index):
		img = torch.from_numpy(np.asarray(self.data[index,:,:,:]))
		label=torch.from_numpy(np.asarray(self.labels[index,:,:]))
		#print("label_shape: ",label.shape)
		# label=torch.from_numpy(np.asarray(raw_label))
		return img, label

	def __len__(self):
		return len(self.labels)

	def __labels__(self):
		return self.labels


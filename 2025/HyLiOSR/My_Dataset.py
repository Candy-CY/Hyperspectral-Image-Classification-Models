
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, lidar_data, labels):
        self.data = data
        self.lidar_data = lidar_data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        lidar_sample = self.lidar_data[idx]
        label = self.labels[idx]
        return data_sample, lidar_sample, label

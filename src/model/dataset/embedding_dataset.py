from torch.utils.data import Dataset
import torch
import os
import glob

class EmbeddingDataset(Dataset):
    def __init__(self, data_path):
        self.files = glob.glob(os.path.join(data_path, "*.pt"))
        self.data = []
        for f in self.files:
            self.data.append(torch.load(f))
        self.data = torch.cat(self.data, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

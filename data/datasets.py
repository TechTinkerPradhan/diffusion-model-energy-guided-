
import torch
from torch.utils.data import Dataset
import numpy as np

class SceneDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        scene = self.data[idx]
        features = scene['features'].astype(np.float32)
        label = scene['label']
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.float32)

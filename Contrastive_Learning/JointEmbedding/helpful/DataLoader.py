import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_sample = self.data[index]
        target = self.targets[index]
        
        # Convert to PyTorch tensors
        data_sample = torch.tensor(data_sample, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.long)
        
        return data_sample, target
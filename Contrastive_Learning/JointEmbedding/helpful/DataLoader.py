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

class FilteredDataset(Dataset):
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = classes
        self.indices = [i for i, label in enumerate(dataset.targets) if label in classes]
        self.targets = [dataset.targets[i] for i in self.indices]
        self.data = [dataset.data[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        label = self.classes.index(label)
        return self.dataset.transform(img), label
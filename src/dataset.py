"""
-------dataset.py-------
This module defines the Dataset class, which is responsible for loading and 
preprocessing the dataset for training and evaluation.
"""

from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch

class MNAddDataset(Dataset):
    def __init__(self, root="./data", train = True, transform = None):
        self.mnist = datasets.MNIST(
            root = root,
            train = train,
            download = True,
            transform = transform or transforms.ToTensor()
        )

    def __len__(self):
        return len(self.mnist) // 2
    
    def __getitem__(self, idx):
        img1, digit1 = self.mnist[2 * idx]
        img2, digit2 = self.mnist[2 * idx + 1]

        x = torch.cat([img1, img2], dim= 2)     #shape = 1 * 28 * 56 (combine two digit images side-by-side isntead of 28 * 28 we have 28 * 56)
        concepts = torch.tensor([digit1,digit2])
        target = (digit1 + digit2) % 2

        return x, concepts, target
    

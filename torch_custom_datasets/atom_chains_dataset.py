from pathlib import Path
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

# Ignore warnings
#import warnings
#warnings.filterwarnings("raise")


class ChainDataset(Dataset):
    
    def __init__(self, path, input_dim=100, transform=None):
        self.path = path
        self.transform = transform

        
        with open(os.path.join(self.path, "Y.npy"), "rb") as file:
            self.y = torch.tensor(np.load(file), dtype=torch.float64).reshape(1, -1, 1)
            
        self.data_paths = self._get_x_paths()
        self.data_dim = (input_dim, input_dim)

    def _get_x_paths(self):
        return [os.path.join(self.path, f"X_{i}.npy") for i in range(1, self.y.shape[1])]
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        mol_path = self.data_paths[idx]
        sample_y = self.y[0, idx]
        
        with open(mol_path, "rb") as f:
            sample = torch.tensor(np.load(f), dtype=torch.float64).reshape(self.data_dim)
        
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, sample_y


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        
        
        # Matrix: Height x Width
        
        # Returns: Channels x Height x Width
        
        return sample.view(1, *sample.shape)
    
class Normalize: # Showed less performance while used
    def __call__(self, sample):
        
        
        new_min = -1
        new_max = 1
        for i in range(sample.shape[0]):
            t_min, t_max = sample[i].min(), sample[i].max()
            
            sample[i] = (sample[i] - t_min)/(t_max - t_min)*(new_max - new_min) + new_min
            
        return sample
            
    
transform=transforms.Compose([ToTensor(),
                              ])
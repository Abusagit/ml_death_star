__doc__ = """Second attempt with neural network learning process of Coulomb law"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import plotly.express as px
import numpy as np

class ChainCNN(nn.Module):
    def __init__(self, input_dim=100, in_channels=1, hidden_dim=32, out_channels=None, dropout=0.25) -> None:
        self.out_channels = out_channels or [5, 10, 15]
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.p = dropout
        
        
        super().__init__()
        
        
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels[0], kernel_size=7, padding=2), # In=100, Out=98 * out_channels[0]
            nn.MaxPool2d(kernel_size=2), # 98 -> 49 * out_channels[0]
            nn.Conv2d(in_channels=self.out_channels[0], out_channels=self.out_channels[1], kernel_size=5, padding=2), # 49 ->  49 * out_channels[1]
            nn.MaxPool2d(kernel_size=7), # 49 -> 7 * out_channels[1]
            nn.Conv2d(in_channels=self.out_channels[1], out_channels=self.out_channels[2], kernel_size=3), #  7 -> 5 * out_channels[2]
            nn.Flatten(), # -> 5 * 5 * out_channels[2] (5 * 5 * 15 = 375)
            nn.Linear(5 * 5 * self.out_channels[2], 128), 
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Linear(128, 32),
            nn.ReLU(), 
            nn.Dropout(self.p / 5),
            nn.Linear(32, 1),
        )
        
        
    def forward(self, x):
        z = self.model(x)
        
        return z
    
    def loss(self, pred, target):
        loss = nn.MSELoss()
        
        return loss(pred, target)
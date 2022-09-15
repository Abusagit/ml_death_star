__doc__ = """Simple graph attention model created with the purpose of searching
whether neural netwoks can learn some simple physics laws and to prove my teacher that it`s [potentially] possible"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
# import torch_geometric.utils as pyg_utils

import plotly.express as px
import numpy as np

class CoulombNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim):
        super(CoulombNet, self).__init__()
                
        self.conv_1 = pyg_nn.GATv2Conv(in_channels=input_dim, out_channels=hidden_dim, heads=2, edge_dim=edge_dim)
        self.head_transform_1 = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim)
        self.batchnorm_1 = nn.BatchNorm1d(hidden_dim)
        
        self.pool_1 = pyg_nn.TopKPooling(in_channels=hidden_dim, ratio=0.9)
        
        self.conv_2 = pyg_nn.GATv2Conv(hidden_dim, hidden_dim, heads=2, edge_dim=edge_dim)
        self.head_transform_2 = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim)
        self.batchnorm_2 = nn.BatchNorm1d(hidden_dim)
        
        self.pool_2 = pyg_nn.TopKPooling(in_channels=hidden_dim, ratio=0.7)
        
        self.conv_3 = pyg_nn.GATv2Conv(hidden_dim, hidden_dim, heads=2, edge_dim=edge_dim)
        self.head_transform_3 = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim)
        self.batchnorm_3 = nn.BatchNorm1d(hidden_dim)
        
        self.pool_3 = pyg_nn.TopKPooling(in_channels=hidden_dim, ratio=0.5)
        
        
        self.linear_1 = nn.Linear(hidden_dim * 3, hidden_dim) # TODO
        self.linear_2 = nn.Linear(hidden_dim, 1)
        
        return
    
    def forward(self, data):
        x, edge_index, edge_attr, y, batch_index = data["x"].float(), data["edge_index"], data["edge_attr"].float(), data["y"].float(), data["batch"]
#         print(data)
#         print(x, edge_index, edge_attr, y, batch, _ptr)
#         print(data['x'])
        
        x = self.conv_1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.head_transform_1(x)
        x = torch.relu(x)
        x = self.batchnorm_1(x)
        
        x, edge_index, edge_attr, batch_index, *_ = self.pool_1(x, edge_index, edge_attr, batch_index)
#         print(x.shape, edge_attr)
        
        x1 = pyg_nn.global_mean_pool(x, batch_index)
        
        x = self.conv_2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.head_transform_2(x)
        x = torch.relu(x)
        x = self.batchnorm_2(x)
        
        x, edge_index, edge_attr, batch_index, *_ = self.pool_2(x, edge_index, edge_attr, batch_index)
        
        x2 = pyg_nn.global_mean_pool(x, batch_index)
        
        x = self.conv_3(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.head_transform_3(x)
        x = torch.relu(x)
        x = self.batchnorm_3(x)
        
        x, edge_index, edge_attr, batch_index, *_ = self.pool_3(x, edge_index, edge_attr, batch_index)
        
        x3 = pyg_nn.global_mean_pool(x,batch_index)
        
        
        x = torch.cat([x1, x2, x3], dim=-1)
        
        x = self.linear_1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear_2(x)
        
        # print(x)
        
        
        return x.flatten()
    
    
    def loss(self, pred, target):
        loss = nn.MSELoss()
        
        
        return loss(pred, target)

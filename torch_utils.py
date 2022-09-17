__doc__ = """Contains various snippets for data preparation, model training and is created for being easy-reusable"""
import numpy as np
import sys
import logging

import torch
from torch_geometric.data import DataLoader as DataloaderGeometric
from torch_geometric.data import Dataset as DatasetGeometric
from torch_geometric.data import Data as DataGeometric
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm

logger = logging.getLogger(__name__)



def get_train_test_dataloaders_geometric(pyg_data, ratio=0.8, random_state=42, batch_size=2):
    torch.manual_seed(random_state)
    dataset = pyg_data.shuffle()
    
    train_samples = int(len(dataset) * ratio)

    train_set = dataset[:train_samples]
    test_set = dataset[train_samples:]
    
    print(f"# of training graphs: {train_samples}\n# of test graphs: {len(dataset) - train_samples}")
    
    train_dataloader = DataloaderGeometric(train_set, batch_size=batch_size, shuffle=True)

    test_dataloader = DataloaderGeometric(test_set, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader

def get_train_test_dataloaders(torch_dataset, ratio=0.8, random_state=42, batch_size=2):
    torch.manual_seed(random_state)
    
    train_num = int(len(torch_dataset) * ratio)
    test_num = len(torch_dataset) - train_num
    
    print(f"# of training elements: {train_num}\n# of validation elements: {test_num}")
    
    trainset, testset = random_split(torch_dataset, [train_num, test_num])
    
    
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return train_dataloader, test_dataloader



def test_geometric(test_loader, model, device, is_validation=False):
    
    model.eval()
    LOSS = []
    torch.cuda.empty_cache()
    for data in test_loader:
        data.to(device)
        predictions = model(data)
        targets = data.y
        
        batch_loss = model.loss(pred=predictions, target=targets).item()
        LOSS.append(batch_loss)
        del data
        torch.cuda.empty_cache()
    return np.mean(LOSS)

def train_geometric(model, optimizer, train_loader, test_loader, optimizer_params, num_of_epochs=200, epoch_interval=10, log=sys.stdout):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    logger.info(f"Model on cuda: {next(model.parameters()).is_cuda}")
    optimizer = optimizer(model.parameters(), **optimizer_params)
    
    train_losses = []
    test_losses = []
    with tqdm(total=num_of_epochs, file=log) as pbar:
        
        # logger.info(f"Started logging to {log}")
        pbar.set_description("Performing training procedure:")
    
        for epoch in range(1, num_of_epochs + 1):
            torch.cuda.empty_cache()
            epoch_losses = []
            model.train()

            for i, batch in enumerate(train_loader):

                batch.to(device)
                optimizer.zero_grad()
                targets = batch.y.float()
                
                
                predictions = model(batch)
                
                loss = model.loss(predictions, targets)
                loss.backward()
                optimizer.step()
                loss.detach()
                train_loss = loss.item()
                epoch_losses.append(train_loss)
                
                logger.debug(f"Epoch {epoch} | batch {i} loss: {train_loss:.4f}")
                
                del batch
                torch.cuda.empty_cache()
            if epoch % epoch_interval == 0:
                test_loss = test_geometric(test_loader, model, device=device)
                test_losses.append(test_loss)
                logger.info(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Test loss : {test_loss:.4f}")
                
                # pbar.write(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Test loss : {test_loss:.4f}")
            
            epoch_loss = np.mean(epoch_losses)
            train_losses.append(epoch_loss)

            
            
    return model, train_losses, test_losses

def test_cnn(test_loader, model, device, is_validation=False):
    
    model.eval()
    LOSS = []
    torch.cuda.empty_cache()
    for (X, targets) in test_loader:
        X = X.to(device).float()
        targets = targets.to(device).float()
        
        predictions = model(X)
        
        
        batch_loss = model.loss(pred=predictions, target=targets).item()
        LOSS.append(batch_loss)
        del X
        torch.cuda.empty_cache()
    return np.mean(LOSS)
    
    
def train_cnn(model, optimizer, train_loader, test_loader, optimizer_params, num_of_epochs=200, epoch_interval=10, log=sys.stdout):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    logger.info(f"Model on cuda: {next(model.parameters()).is_cuda}")
    optimizer = optimizer(model.parameters(), **optimizer_params)
    
    train_losses = []
    test_losses = []
    with tqdm(total=num_of_epochs, file=log) as pbar:
        
        # logger.info(f"Started logging to {log}")
        pbar.set_description("Performing training procedure:")
    
        for epoch in range(1, num_of_epochs + 1):
            torch.cuda.empty_cache()
            epoch_losses = []
            model.train()

            for i, (X, targets) in enumerate(train_loader):

                X, targets = X.to(device).float(), targets.to(device).float()
                
                optimizer.zero_grad()
                
                predictions = model(X)
                
                loss = model.loss(predictions, targets)
                loss.backward()
                optimizer.step()
                loss.detach()
                train_loss = loss.item()
                epoch_losses.append(train_loss)
                
                logger.debug(f"Epoch {epoch} | batch {i} loss: {train_loss:.4f}")
                
                del batch
                torch.cuda.empty_cache()
            if epoch % epoch_interval == 0:
                test_loss = test_cnn(test_loader, model, device=device)
                test_losses.append(test_loss)
                logger.info(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Test loss : {test_loss:.4f}")
                
                # pbar.write(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Test loss : {test_loss:.4f}")
            
            epoch_loss = np.mean(epoch_losses)
            train_losses.append(epoch_loss)

            
            
    return model, train_losses, test_losses
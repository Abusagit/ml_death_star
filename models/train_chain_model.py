import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch_geometric.nn as pyg_nn
import plotly.express as px
import pickle
# import numpy as np

import argparse
import logging

import sys
import os

from chain_atom_model import ChainCNN
from ml_death_star.torch_utils import get_train_test_dataloaders, train
from ml_death_star.torch_custom_datasets.atom_chains_dataset import ChainDataset, ToTensor

# from torch_geometric.data import DataLoader as DataloaderGeometric
# from torch_geometric.data import Dataset as DatasetGeometric

# from tqdm import tqdm
from pathlib import Path


def get_parser():
    root = argparse.ArgumentParser("Training CoulombNet for investigating NN capability of reinventing Coulomb`s law")
    
    root.add_argument("--dataroot", type=Path, help="Directory with Pytorch dataset")
    root.add_argument("--train_test_ratio", type=float, default=0.8)
    root.add_argument("--batch_size", type=int, default=40)
    root.add_argument("--hidden_dim", type=int, default=10)
    root.add_argument("--epochs", default=200, type=int)
    root.add_argument("-o", "--outdir", default=Path.cwd())
    
    return root

def get_logger(outdir):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(str(Path(outdir, "training.log")), mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
    
SUCCESS_MSG = f"{'=' * 20} DONE {'=' * 20}"

# Parsing arguments
parser = get_parser()
args = parser.parse_args()

Path(args.outdir).mkdir(parents=True, exist_ok=True)
# Initialise logging
logger = get_logger(outdir=args.outdir)
logger.info(f"Logger initialised. Logs are stored at {Path(args.outdir, 'training.log')}")
logger.info(f"Arguments are: {args}")

# defining dataset
logger.info(f"Initialising dataset at {args.dataroot}")


dataset = ChainDataset(path=args.dataroot, 
                       transform=ToTensor)

train_dl, test_dl = get_train_test_dataloaders(dataset, 
                                               ratio=args.train_test_ratio,
                                               batch_size=args.batch_size,
                                            )

logger.info(SUCCESS_MSG)


logger.info("Initialising model and starting training")
logger.info(f"Cuda available: {torch.cuda.is_available()=}")

optimizer = torch.optim.Adam
optimizer_params = {"lr": 0.01}
torch.cuda.empty_cache()
model, train_losses, test_losses = train(model=ChainCNN(input_dim=100, hidden_dim=args.hidden_dim),
                                        optimizer=optimizer,
                                        optimizer_params=optimizer_params,
                                        log=open(os.devnull,"w"),
                                        num_of_epochs=args.epochs,
                                        train_loader=train_dl,
                                        test_loader=test_dl,
                                        )

torch.save(model.state_dict(), args.outdir / "params.pkl")

with open(Path(args.outdir) / "train.pkl", "wb") as h:
    pickle.dump(train_losses, h, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(Path(args.outdir) / "test.pkl", "wb") as h:
    pickle.dump(test_losses, h, protocol=pickle.HIGHEST_PROTOCOL)

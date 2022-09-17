import torch
import plotly.express as px
import pickle

import argparse
import logging

import sys
import os

from chain_atom_model import ChainCNN
from ml_death_star.torch_utils import get_train_test_dataloaders, train_cnn
from ml_death_star.torch_custom_datasets.atom_chains_dataset import ChainDataset, transform

from pathlib import Path


def get_parser():
    root = argparse.ArgumentParser("Training CoulombNet for investigating NN capability of reinventing Coulomb`s law")
    
    root.add_argument("--dataroot", type=Path, help="Directory with Pytorch dataset")
    root.add_argument("--train_test_ratio", type=float, default=0.8)
    root.add_argument("--batch_size", type=int, default=40)
    root.add_argument("--hidden_dim", type=int, default=10)
    root.add_argument("--input_dim", type=int, default=100)
    root.add_argument("--epochs", default=200, type=int)
    root.add_argument("-o", "--outdir", default=os.getcwd())
    
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
                       input_dim=args.input_dim,
                       transform=transform)

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
model, train_losses, test_losses = train_cnn(model=ChainCNN(input_dim=args.input_dim, hidden_dim=args.hidden_dim),
                                        optimizer=optimizer,
                                        optimizer_params=optimizer_params,
                                        log=open(os.devnull,"w"),
                                        num_of_epochs=args.epochs,
                                        train_loader=train_dl,
                                        test_loader=test_dl,
                                        )

torch.save(model.state_dict(), args.outdir / "params.pkl")

with open(Path(args.outdir, "train.pkl"), "wb") as h:
    pickle.dump(train_losses, h, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(Path(args.outdir, "test.pkl"), "wb") as h:
    pickle.dump(test_losses, h, protocol=pickle.HIGHEST_PROTOCOL)

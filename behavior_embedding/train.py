import os
import hydra
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from dataset import Dataset
from embedding import BehaviorEmbedding
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger

def set_seed(seed):
    pl.seed_everything(seed)

def set_device(device):
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('medium')

@hydra.main(config_name="config_embed")  
def main(cfg):

    set_seed(cfg.seed)
    set_device(cfg.device)

    run_name = f"traj-{cfg.train.optimizer.lr}-{cfg.seed}-{cfg.data.batch_size}"
    wandb.init(project="policy_generator", name=run_name) 

    datamodule = Dataset(cfg.data) 
    normalizer = datamodule.get_normalizer()

    system = BehaviorEmbedding(cfg) 
    system.set_normalizer(normalizer)
    trainer: Trainer = hydra.utils.instantiate(cfg.train.trainer)
    wandb_logger = WandbLogger()
    trainer.logger = wandb_logger

    # Train the model
    trainer.fit(system, datamodule=datamodule) 
    wandb.finish()
if __name__ == "__main__":
    main()

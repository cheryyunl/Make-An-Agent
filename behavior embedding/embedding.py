import numpy as np
from base_system import BaseSystem
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pdb
import wandb
import hydra
from embed_model import Embedding
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from normalizer import LinearNormalizer

class BehaviorEmbedding(BaseSystem):
    def __init__(self, config, **kwargs):
        super(BehaviorEmbedding, self).__init__(config)
        print("Behavior Embedding init")
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.train_cfg = config.train
        self.model_cfg = config.model
        self.model = self.build_model()
        self.batch_size = config.data.batch_size
        self.normalizer = LinearNormalizer() 
    
    def set_normalizer(self, normalizer: LinearNormalizer):
       self.normalizer.load_state_dict(normalizer.state_dict())
    
    def build_model(self, **kwargs):
        model = Embedding(self.model_cfg)
        return model
        
    def configure_optimizers(self, **kwargs):
        parameters = self.model.parameters()
        self.optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, parameters)

        self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

        return self.optimizer

    def training_step(self, batch, **kwargs):
        optimizer = self.optimizers()
        loss = self.forward(batch, **kwargs)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()
        wandb.log({"train/loss": loss})
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, **kwargs):
        batch = self.normalizer.normalize(batch)
        traj_e = self.model.traj_embedding(batch['traj']) 
        task_e = self.model.task_embedding(batch['task'])
        val_loss = self.embed_loss(traj_e, task_e) 
        wandb.log({"val/loss": val_loss.detach()})
        self.log('val_loss', val_loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': val_loss}
    

    def forward(self, batch, **kwargs):
        batch = self.normalizer.normalize(batch)
        traj_e, task_e = self.model(batch['traj'], batch['task'])
        loss = self.embed_loss(traj_e, task_e)
        self.log('loss', loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def embed_loss(self, traj_e, task_e):
        logits = self.model.compute_logits(traj_e, task_e)
        labels = torch.arange(logits.shape[0]).long().to(traj_e.device)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, labels)
        return loss





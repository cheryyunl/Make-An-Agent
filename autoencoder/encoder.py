import wandb
import pdb
import hydra

import numpy as np
from base_system import BaseSystem
import torch.nn as nn
import pytorch_lightning as pl
from model_proj import EncoderDecoder
import torch.optim.lr_scheduler
from typing import Optional, Union, List, Dict, Any, Sequence
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts



class Encoder(BaseSystem):
    def __init__(self, cfg):
        super(Encoder, self).__init__(cfg)
        print("Encoder init")
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.train_cfg = cfg.train
        self.model_cfg = cfg.model
        self.model = self.build_model()
        self.loss_func = self.build_loss_func()

    def training_step(self, batch, **kwargs):
        optimizer = self.optimizers()
        param, traj, task = batch
        loss = self.forward(param, **kwargs)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()
        wandb.log({"train/loss": loss})
        return {'loss': loss}

    def build_model(self, **kwargs):
        self.model = EncoderDecoder(self.model_cfg.n_embd,
                                    self.model_cfg.encoder_depth,
                                    self.model_cfg.decoder_depth,
                                    self.model_cfg.input_noise_factor,
                                    self.model_cfg.latent_noise_factor)
        if self.train_cfg.finetune:
            self.load_encoder(self.train_cfg.pretrain_model)
        return self.model

    def build_loss_func(self):
        if 'loss_func' in self.train_cfg:
            loss_func = hydra.utils.instantiate(self.train_cfg.loss_func)
            return loss_func

    def configure_optimizers(self, **kwargs):
        parameters = self.model.parameters()
        self.optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, parameters)

        self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

        return self.optimizer

    def validation_step(self, batch, batch_idx, **kwargs):
        param, traj, task = batch
        embed = self.model.encode(param) 
        outputs = self.model.decode(embed)
        val_loss = self.loss_func(outputs, param) 
        wandb.log({"val/loss": val_loss.detach()})
        self.log('val_loss', val_loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': val_loss}


    def forward(self, batch, **kwargs):
        output = self.model(batch)
        loss = self.loss_func(batch, output, **kwargs)
        self.log('loss', loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def encode(self, x, **kwargs):
        return self.model.encode(x)

    def decode(self, x, **kwargs):
        return self.model.decode(x)

    def load_encoder(self, encoder_path, evaluate=False):
        print("Loading encoders from {}".format(encoder_path))
        encoder_ckpt = torch.load(encoder_path, map_location='cpu')
        weights_dict = {}
        weights = encoder_ckpt['state_dict']
        for k, v in weights.items():
            new_k = k.replace('model.', '') if 'model.' in k else k
            weights_dict[new_k] = v
        self.model.load_state_dict(weights_dict)

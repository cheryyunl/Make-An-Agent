import numpy as np
from base_system import BaseSystem
import torch.nn as nn
import pytorch_lightning as pl
import pdb


class Encoder(BaseSystem):
    def __init__(self, config, **kwargs):
        super(Encoder, self).__init__(config)
        print("Encoder init")
        self.save_hyperparameters()

    def forward(self, batch, **kwargs):
        output = self.model(batch)
        loss = self.loss_func(batch, output, **kwargs)
        self.log('loss', loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def encode(self, x, **kwargs):
        return self.model.encode(x)

    def decode(self, x, **kwargs):
        return self.model.decode(x)

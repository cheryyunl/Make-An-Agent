import os
import torch
from torch.utils.data import Dataset, random_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

class Dataset(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = getattr(cfg, 'batch_size')
        self.num_workers = getattr(cfg, 'num_workers')
        self.data_root = getattr(self.cfg, 'data_root')
        print("data root", self.data_root)

        # Check the root path
        # assert os.path.exists(self.data_root), f'{self.data_root} not exists'

        state = torch.load(self.data_root)
        self.params = state['param']
        self.task = state['task']
        self.traj = state['traj']
        self.setup

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        param = self.params[idx]
        traj = self.traj[idx]
        task = self.task[idx]
        return param, traj, task

    @property
    def setup(self):
        train_size = int(0.9 * len(self))
        val_size = int(0.09 * len(self))
        test_size = test_size = len(self) - train_size - val_size
        print("train_size:", train_size)
        print("val_size:", val_size)
        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self, [train_size, val_size, test_size]
        )


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)

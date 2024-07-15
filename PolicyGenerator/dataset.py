import os
import torch
from torch.utils.data import Dataset, random_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from policy_generator.model.common.normalizer import LinearNormalizer

class Dataset(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = getattr(cfg, 'batch_size', 128)
        self.num_workers = getattr(cfg, 'num_workers', 4)
        self.data_root = getattr(self.cfg, 'data_root')
        # Check the root path
        assert os.path.exists(self.data_root), f'{self.data_root} not exists'
        self.data = torch.load(self.data_root, map_location='cpu')
        if len(self.data['param'].shape) == 2:
            self.data['param'] = self.data['param'].reshape(-1, 2, 1024)
        self.setup

    def __len__(self):
        return len(self.data['task'])

    def __getitem__(self, idx):
        params = self.data['param'][idx]
        traj = self.data['traj'][idx]
        task = self.data['task'][idx]

        return {'param': params, 'traj': traj, 'task': task} 

    @property
    def setup(self):
        train_size = int(0.9 * len(self))
        val_size = int(0.09 * len(self))
        test_size = len(self) - train_size - val_size

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
    

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'param': self.data['param'],
            'traj': self.data['traj'],
            'task': self.data['task']}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

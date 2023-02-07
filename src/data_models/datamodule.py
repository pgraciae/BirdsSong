from pathlib import Path
from typing import Dict

import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch

from .dataset import BirdDataset


class BirdDataModule(pl.LightningDataModule):

    def __init__(self, root_data_dir = './data', batch_size = 32, num_workers = 4,
                 transforms = {'transform' : None, 'target_transform' : None}, seed = 0, test_size = 0.2):
        super().__init__()

        self.root_data_dir = Path(root_data_dir).resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = BirdDataset
        self.transform = transforms['transform']
        self.target_transform = transforms['target_transform']
        self.seed = torch.Generator().manual_seed(seed)
        self.test_size = test_size
        self.coder = 0

    def prepare_data(self):
        pass
    
    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            dataset = self.dataset(data_path = self.root_data_dir, train = True, transform = self.transform, target_transform=self.target_transform)
            val_len = int(len(dataset) * self.test_size)
            #add labelencoder
            self.coder = dataset.coder

            self.bird_train, self.bird_val = random_split(dataset, [len(dataset) - val_len, val_len], generator = self.seed)

        if stage == 'test' or stage is None:
            self.bird_test = self.dataset(data_path = self.root_data_dir, train = False, transform = self.transform, target_transform=self.target_transform)
        
    def train_dataloader(self):
        return DataLoader(self.bird_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.bird_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.bird_test, batch_size=self.batch_size, num_workers=self.num_workers)
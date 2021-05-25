import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
import numpy as np
import os
from torch.nn import functional as F
import numpy as np
from typing import Dict
import copy
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import math
from utils import summarize_prune
from provided_code.datasource import DataLoaders
from pytorch_lightning.metrics import functional as FM
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, BatchSampler
from pytorch_lightning.callbacks import GPUStatsMonitor

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset1 = MNIST('./data', train=True,
                 transform=transform)
dataset2 = MNIST('./data', train=False,
                 transform=transform)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output


class Model(pl.LightningModule):
    def __init__(
        self,
        batch_size=32,
        lr=1e-3,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.model = Net()
        self.lr = lr
        self.batch_size = batch_size,

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=self.lr, params=self.model.parameters())
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        metrics = {'loss': loss, 'acc': acc}
        self.log_dict(metrics)
        return metrics


model = Model()
logger = TensorBoardLogger(save_dir="./logs", name='trial')

train_loader = DataLoader(dataset1, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset2, batch_size=32, shuffle=False, num_workers=4)

trainer = pl.Trainer(gpus=1,
                     progress_bar_refresh_rate=20,
                     profiler="simple",
                     callbacks=[GPUStatsMonitor(
                         fan_speed=True, temperature=True)],
                     max_epochs=20)
trainer.fit(model=model,
            train_dataloader=train_loader,
            val_dataloaders=test_loader)




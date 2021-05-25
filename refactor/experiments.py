import os
import time
import wandb
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger as Logger


def experiment(args):
    logger = Logger(project=args.exp_name, save_dir=args.log_dir)


# .item(), .numpy(), .cpu()
# Don’t call .item() anywhere in your code.
# Use .detach() instead to remove the connected graph calls.
# Lightning takes a great deal of care to be optimized for this.

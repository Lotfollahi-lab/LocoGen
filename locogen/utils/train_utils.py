import torch
import random
import numpy as np
import os

from locogen.datasets.enhancer_gosai_dataset import get_dataloaders_gosai

def load_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataloader(config):
    if config.dataset.name == 'gosai':
        return get_dataloaders_gosai(config)
    raise ValueError(f"Unknown dataset: {config.dataset.name}")

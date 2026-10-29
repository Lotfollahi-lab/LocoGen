# Adapted from https://github.com/ChenyuWang-Monica/DRAKES/blob/master/drakes_dna/dataloader_gosai.py

import torch
import pandas as pd
import numpy as np
import os

DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3} #, 'M': 4}

class GosaiDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data_df = pd.read_csv(data_path)
        self.seqs = torch.tensor(data_df['seq'].apply(lambda x: [DNA_ALPHABET[c] for c in x]).tolist())
        self.clss = torch.tensor(data_df[['hepg2', 'k562', 'sknsh']].to_numpy())

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {'seqs': self.seqs[idx], 'clss': self.clss[idx], 'attention_mask': torch.ones(len(self.seqs[idx]))}

def get_dataloaders_gosai(config, skip_valid=False, valid_seed=42):
    num_gpus = max(torch.cuda.device_count(), 1)  # CPU-only -> treat as one device
    if config.loader.global_batch_size % (num_gpus * config.train.accumulate_grad_batches) != 0:
        raise ValueError(
           f'Train batch size {config.loader.global_batch_size} not divisible by '
           f'{num_gpus} gpus with accumulation '
           f'{config.train.accumulate_grad_batches}.')
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
           f'Eval batch size {config.loader.eval_global_batch_size} '
           f'not divisible by {num_gpus} gpus.')
    full_set = GosaiDataset(config.dataset.data_path)
    N = len(full_set)

    # One seeded permutation produces disjoint train / val / test index sets.
    # Previously each split was drawn with an independent np.random.choice from
    # the full dataset, which (a) used the uncontrolled global NumPy RNG for
    # val/test and (b) allowed heavy overlap between train and the evaluation
    # splits — invalidating held-out metrics.
    split_seed = int(config.dataset.get("split_seed", 42))
    val_size   = int(config.dataset.get("val_size",   40000))
    test_size  = int(config.dataset.get("test_size",  40000))
    if val_size + test_size >= N:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) must be < "
            f"len(dataset) ({N})"
        )
    perm = np.random.RandomState(split_seed).permutation(N)
    test_idx  = perm[:test_size]
    valid_idx = perm[test_size : test_size + val_size]
    train_idx = perm[test_size + val_size :]

    train_cap = config.dataset.get("train_size", None)
    if train_cap is not None and int(train_cap) < len(train_idx):
        train_idx = train_idx[: int(train_cap)]

    train_set = torch.utils.data.Subset(full_set, train_idx)
    valid_set = torch.utils.data.Subset(full_set, valid_idx)
    test_set  = torch.utils.data.Subset(full_set, test_idx)
    print(f"[gosai] split sizes: train={len(train_set)}  "
          f"val={len(valid_set)}  test={len(test_set)}  "
          f"(disjoint, split_seed={split_seed}, full={N})")

    persistent = config.loader.num_workers > 0  # DataLoader rejects persistent_workers with 0 workers
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=config.loader.batch_size,
                                               num_workers=config.loader.num_workers,
                                               pin_memory=config.loader.pin_memory,
                                               shuffle=not config.dataset.streaming,
                                               persistent_workers=persistent)
    if skip_valid:
        return train_loader, None, None

    if valid_seed is None:
        shuffle_valid = False
        generator = None
    else:
        shuffle_valid = True
        generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=config.loader.eval_batch_size,
                                               num_workers=config.loader.num_workers,
                                               pin_memory=config.loader.pin_memory,
                                               shuffle=shuffle_valid, generator=generator)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=config.loader.eval_batch_size,
                                              num_workers=config.loader.num_workers,
                                              pin_memory=config.loader.pin_memory,
                                              shuffle=shuffle_valid, generator=generator)
    return train_loader, valid_loader, test_loader

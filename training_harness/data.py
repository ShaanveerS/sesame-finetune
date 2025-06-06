from datasets import Dataset, load_from_disk
from functools import partial
from moshi.models import MimiModel
import torch
from torch.utils.data import DataLoader
from .config import TrainingConfig
from typing import Optional, Tuple
import pandas as pd
import os


def load_mimi_ds(config: TrainingConfig) -> Tuple[Dataset, Dataset]:
    ds = load_from_disk(config.dataset.dataset_dir)
    if "test" not in ds:
        ds = ds["train"].train_test_split(test_size=1)  # Split with 1 example for test set
    
    # ignore training val split for now
    return ds["train"], ds["test"]


def collate_fn_mse(batch, mimi_model: MimiModel, codebook_size: int = 32):
    """
    batch is a list of dicts: each dict has "tokens" shape [depth+1, T],
    and "labels" shape [depth+1, T].
    We pad them into [B, depth+1, T_max].
    """
    quantizer = mimi_model.quantizer.acoustic_quantizer.vq

    height = codebook_size + 1
    max_input_len = max(item["ground_truth"].shape[0] - 1 for item in batch)

    B = len(batch)
    tokens = torch.full((B, max_input_len, height), 0, dtype=torch.long)  # 2=some <PAD>
    tokens_masks = torch.full((B, max_input_len, height), 0, dtype=torch.bool)
    pad_mask = torch.full((B, max_input_len), False, dtype=torch.bool)

    targets = torch.full((B, max_input_len, 256), 0, dtype=torch.float32)

    for i, item in enumerate(batch):
        seq_len = item["ground_truth"].shape[0] - 1
        tokens[i, :seq_len, :] = item["ground_truth"][:-1, :].clone()
        tokens_masks[i, :seq_len, :] = item["ground_truth_masks"][:-1, :].clone()
        pad_mask[i, :seq_len] = True

        label = item["ground_truth"][1:, :]
        # full block of zeros for audio codes
        acoustic_codes = label[:, 1:-1].T
        final_residuals = quantizer.decode(acoustic_codes.unsqueeze(-1)).squeeze(-1)
        # zero text positions with the mask
        mask = item["ground_truth_masks"][1:, :-1].all(dim=1)
        final_residuals[~mask] = 0
        targets[i, :seq_len, :] = final_residuals.unsqueeze(0)

    return {"tokens": tokens, "targets": targets, "tokens_masks": tokens_masks, "pad_mask": pad_mask}

def collate_fn(batch, codebook_size: int = 32):
    """
    batch is a list of dicts: each dict has "tokens" shape [depth+1, T],
    and "labels" shape [depth+1, T].
    We pad them into [B, depth+1, T_max].

    """
    height = codebook_size + 1
    max_input_len = max(item["ground_truth"].shape[0] for item in batch) - 1

    B = len(batch)
    tokens = torch.full((B, max_input_len, height), 0, dtype=torch.long)  # 2=some <PAD>
    tokens_masks = torch.full((B, max_input_len, height), 0, dtype=torch.bool)
    acoustic_codes = torch.full((B, max_input_len, height - 2), 0, dtype=torch.long)

    # CSM does not model text
    # Set all positions to MASK by default
    labels = torch.full((B, max_input_len, codebook_size), -100, dtype=torch.long)
    pad_mask = torch.full((B, max_input_len), False, dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = item["ground_truth"].shape[0] - 1
        tokens[i, :seq_len, :] = item["ground_truth"][:-1, :].clone()
        tokens_masks[i, :seq_len, :] = item["ground_truth_masks"][:-1, :].clone()
        pad_mask[i, :seq_len] = True
        # NOTE! The acoustic codes are for step $n+1$ rather than $n$, and omit step 32.
        # This is because the temporal transformer is causal _within_ not across a global timestep.
        acoustic_codes[i, :seq_len, :] = item["ground_truth"][1:, :-2].clone()

        # Predict audio codes causally shifted
        shifted = item["ground_truth"][1:, :-1].clone()
        mask = item["ground_truth_masks"][1:, :-1].all(dim=1)
        shifted[~mask] = -100
        labels[i, :seq_len, :] = shifted

    return {"tokens": tokens, "tokens_masks": tokens_masks, "acoustic_codes": acoustic_codes, "labels": labels, "pad_mask": pad_mask}



def create_dataloaders(config: TrainingConfig, mimi_model: Optional[MimiModel], is_shortcut: bool = False) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    train_ds, val_ds = load_mimi_ds(config)

    collate = partial(collate_fn_mse, mimi_model=mimi_model, codebook_size=32) if is_shortcut else collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

from datasets import Dataset, load_from_disk
from moshi.models import MimiModel
import torch
from torch.utils.data import DataLoader
from train.config import TrainingConfig
from typing import Tuple


def load_mimi_ds(config: TrainingConfig) -> Tuple[Dataset, Dataset]:
    # TODO generalize this for something else
    ds = load_from_disk(config.dataset_dir)
    # ignore training val split for now
    return ds["train"], ds["test"]


def collate_fn(batch, mimi_model: MimiModel, codebook_size: int = 32):
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
    targets = torch.full((B, 256, max_input_len), 0, dtype=torch.float32)

    pad_mask = torch.ones(B, max_input_len)

    for i, item in enumerate(batch):
        seq_len = item["ground_truth"].shape[0] - 1
        tokens[i, :seq_len, :] = item["ground_truth"][:-1, :].clone()

        label = item["ground_truth"][1:, :]
        # full block of zeros for audio codes
        codes = label[:, :-1]
        final_residuals = quantizer.dequantize(codes)
        # zero text positions with the mask
        mask = batch["ground_truth_masks"][1:, :-1].all(dim=1)
        final_residuals[~mask] = 0
        








        # Get full dequantized codes


        

        labels[i, :seq_len, :] = label

        # bookkeeping for ragged batches
        pad_mask[i, :seq_len] = False

    return {"tokens": tokens, "labels": labels, "pad_mask": pad_mask}


def create_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    train_ds, val_ds = load_mimi_ds(config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

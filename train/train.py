from einops import rearrange
import torch
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
import torch.functional as F
from train.config import TrainingConfig
from typing import List

from modeling.generator import Generator


def step(gen: Generator, config: TrainingConfig, optim: AdamW):
    pass



def train_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    gradient_clip: float = 0.0,
) -> TrainStepOutput:
    """Single training step"""
    tokens = batch["tokens"].to(device)
    labels = batch["labels"].to(device)
    pad_mask = batch["pad_mask"].to(device)

    outputs = model(inp=tokens, key_padding_mask=pad_mask)
    base_loss, semantic_loss, _ = compute_losses_mse(outputs, labels)
    loss = base_loss + semantic_loss

    optimizer.zero_grad()
    loss.backward()
    # time.sleep(0.05)
    if gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    scheduler.step()

    return TrainStepOutput(
        loss=loss,
        base_loss=base_loss.item(),
        semantic_loss=semantic_loss.item(),
        lr=scheduler.get_last_lr()[0],
    )


def train(config: TrainingConfig, train_loader: DataLoader, val_loader: DataLoader):
    pass

import torch
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
import torch.functional as F
from training_harness.config import TrainingConfig
from typing import NamedTuple

from modeling.shortcut_layer import ShortcutLayer


class TrainStepOutput(NamedTuple):
    loss: float
    lr: float

def compute_losses_mse(outputs, targets, tokens_masks):
    semantic_loss = F.mse_loss(outputs, targets, reduction="none")
    # Zero out non-acoustic tokens
    acoustic_mask = tokens_masks[:, :, 1:-1]
    semantic_loss[~acoustic_mask] = 0
    # Scale by number of non-masked tokens rather than just the mean
    loss = torch.sum(semantic_loss) / torch.sum(acoustic_mask)
    return loss

# ef compute_losses_logits(logits, targets, tokens_masks):


def train_step(
    model: torch.nn.Module,
    shortcut: ShortcutLayer,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    gradient_clip: float = 0.0,
    shortcut_idx: int = 16,
) -> TrainStepOutput:
    """Single training step"""
    tokens = batch["tokens"].to(device)
    tokens_masks = batch["tokens_masks"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    # targets = batch["targets"].to(device)
    labels = batch["labels"].to(device)

    # b s n d
    acoustic_hidden_states = model(tokens=tokens, tokens_mask=tokens_masks, key_padding_mask=pad_mask)

    shortcut_hidden_states = acoustic_hidden_states[:, :, shortcut_idx, :]
    # TODO do i need to squeeze n?
    outputs = shortcut(shortcut_hidden_states)

    loss = compute_losses_mse(outputs, targets, tokens_masks)

    optimizer.zero_grad()
    loss.backward()
    # time.sleep(0.05)
    if gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    scheduler.step()

    return TrainStepOutput(
        loss=loss,
        lr=scheduler.get_last_lr()[0],
    )


def train(config: TrainingConfig, train_loader: DataLoader, val_loader: DataLoader, model: torch.nn.Module, shortcut: ShortcutLayer, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler):

    device = torch.device("cuda")
    for epoch in range(config.num_epochs):
        for batch in train_loader:
            print(batch.keys())
            train_step(model, shortcut, batch, optimizer, scheduler, device, config.optim.gradient_clip)


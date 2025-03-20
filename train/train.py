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


def compute_losses_mse(
    outputs, labels: torch.Tensor, per_codebook_loss: bool = False
) -> tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Compute base and semantic losses, plus individual codebook losses"""
    # Base loss computation remains the same
    base_loss = F.cross_entropy(
        outputs.token_logits.view(-1, outputs.token_logits.size(-1)),
        labels[:, 0, :].reshape(-1),
        ignore_index=-100,
    )

    # Compute individual codebook losses
    n_codebooks = labels.shape[1] - 1  # Subtract 1 for the base tokens
    if per_codebook_loss:
        codebook_losses = []

        for i in range(n_codebooks):
            # Reshape logits and labels for current codebook
            current_logits = outputs.codebook_logits[:, :, i, :]  # [batch, seq, vocab]
            current_labels = labels[:, i + 1, :]  # [batch, seq]

            loss = F.cross_entropy(
                current_logits.reshape(-1, current_logits.size(-1)),
                current_labels.reshape(-1),
                ignore_index=-100,
            )
            codebook_losses.append(loss.item())
    else:
        codebook_losses = []

    # Compute total semantic loss (same as before, just using einops)
    codebook_logits = rearrange(outputs.codebook_logits, "b s n d -> (b s n) d")
    codebook_labels = rearrange(labels[:, 1:, :], "b n s -> (b s n)")
    semantic_loss = F.cross_entropy(codebook_logits, codebook_labels, ignore_index=-100)

    return base_loss, semantic_loss, codebook_losses


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

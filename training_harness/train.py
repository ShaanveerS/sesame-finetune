import torch
from einops import rearrange
from torch.utils.data import DataLoader
import torch.nn.functional as F
from training_harness.config import TrainingConfig
from typing import NamedTuple, Optional
from tqdm import tqdm

from modeling.shortcut_layer import ShortcutLayer


class TrainStepOutput(NamedTuple):
    loss: float

def compute_losses_mse(outputs, targets, tokens_masks):
    semantic_loss = F.mse_loss(outputs, targets, reduction="none")
    # Zero out non-acoustic tokens
    acoustic_mask = tokens_masks[:, :, 1:-1]
    semantic_loss[~acoustic_mask] = 0
    # Scale by number of non-masked tokens rather than just the mean
    loss = torch.sum(semantic_loss) / torch.sum(acoustic_mask)
    return loss

def compute_losses_logits(all_logits, labels, compute_amortize_mask):
    if compute_amortize_mask is not None:
        labels = labels.masked_fill(compute_amortize_mask.unsqueeze(-1), -100)

    # for better or worse, loss is on audio only
    codebook_labels = rearrange(labels, "b s n -> (b s n)")
    codebook_logits = rearrange(all_logits, "b s n d -> (b s n) d")

    # TODO consider weighting code0 loss more
    loss = F.cross_entropy(
        codebook_logits,
        codebook_labels,
        ignore_index=-100,
    )
    return loss


def train_step(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    accumulate_step: int = 1,
    shortcut: Optional[ShortcutLayer] = None,
    shortcut_idx: int = 16,
) -> TrainStepOutput:
    """Single training step"""
    tokens = batch["tokens"].to(device)
    tokens_masks = batch["tokens_masks"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    # targets = batch["targets"].to(device)
    labels = batch["labels"].to(device)
    acoustic_codes = batch["acoustic_codes"].to(device)

    codebook_logits, compute_amortize_mask = model(tokens=tokens, tokens_mask=tokens_masks, acoustic_codes=acoustic_codes, key_padding_mask=pad_mask)

    # shortcut_hidden_states = acoustic_hidden_states[:, :, shortcut_idx, :]
    # # TODO do i need to squeeze n?
    # outputs = shortcut(shortcut_hidden_states)
    # loss = compute_losses_mse(outputs, targets, tokens_masks)
    loss = compute_losses_logits(codebook_logits, labels, compute_amortize_mask)
    loss = loss / accumulate_step

    loss.backward()
    return TrainStepOutput(
        loss=loss,
    )


def train(config: TrainingConfig, train_loader: DataLoader, val_loader: DataLoader, model: torch.nn.Module, shortcut: ShortcutLayer, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler):
    global_step = 0
    accumulate_step = 0
    device = torch.device("cuda")
    for epoch in range(config.num_epochs):
        pbar = tqdm(train_loader)
        for batch in pbar:
            output = train_step(
                model,
                batch=batch,
                device=device,
                accumulate_step=config.optim.accumulate_steps,
            )
            global_step += 1
            accumulate_step += 1

            if accumulate_step == config.optim.accumulate_steps:
                if config.optim.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.gradient_clip)
                optimizer.zero_grad()
                optimizer.step()
                accumulate_step = 0
                # TODO delete this, I have low VRAM
                torch.cuda.empty_cache()

            lr = scheduler.get_last_lr()[0]
            pbar.set_description(f"Epoch {epoch}, Step {global_step}, Loss {output.loss:.4f}, LR {lr:.6f}")

            if global_step == 100:
                raise ValueError("TEST")

        # TODO move this to step level
        scheduler.step()
    return global_step

        

                


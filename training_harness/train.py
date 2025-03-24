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
    code0_loss: float
    acoustic_loss: float

def compute_losses_mse(outputs, targets, tokens_masks):
    semantic_loss = F.mse_loss(outputs, targets, reduction="none")
    # Zero out non-acoustic tokens
    acoustic_mask = tokens_masks[:, :, 1:-1]
    semantic_loss[~acoustic_mask] = 0
    # Scale by number of non-masked tokens rather than just the mean
    loss = torch.sum(semantic_loss) / torch.sum(acoustic_mask)
    return loss

from dataclasses import dataclass

@dataclass
class LossComponents:
    code0_loss: torch.Tensor
    acoustic_loss: torch.Tensor
    total_loss: torch.Tensor

def compute_losses_logits(all_logits, labels, compute_amortize_mask):
    if compute_amortize_mask is not None:
        labels = labels.masked_fill(compute_amortize_mask.unsqueeze(-1), -100)

    code0 = labels[:, :, 0]
    code0_logits = all_logits[:, :, 0, :]
    acoustic_labels = labels[:, :, 1:]
    acoustic_logits = all_logits[:, :, 1:, :]

    code0 = rearrange(code0, "b s -> (b s)")
    code0_logits = rearrange(code0_logits, "b s d -> (b s) d")
    acoustic_labels = rearrange(acoustic_labels, "b s n -> (b s n)")
    acoustic_logits = rearrange(acoustic_logits, "b s n d -> (b s n) d")

    code0_loss = F.cross_entropy(code0_logits, code0, ignore_index=-100)
    acoustic_loss = F.cross_entropy(acoustic_logits, acoustic_labels, ignore_index=-100)
    total_loss = code0_loss + acoustic_loss

    return LossComponents(code0_loss, acoustic_loss, total_loss)

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
    loss_components = compute_losses_logits(codebook_logits, labels, compute_amortize_mask)
    loss = loss_components.total_loss / accumulate_step

    loss.backward()
    return TrainStepOutput(
        loss=loss,
        code0_loss=loss_components.code0_loss,
        acoustic_loss=loss_components.acoustic_loss,
    )


def train(config: TrainingConfig, train_loader: DataLoader, val_loader: DataLoader, model: torch.nn.Module, shortcut: ShortcutLayer, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler):
    global_step = 0
    accumulate_step = 0

    # Logging
    step_loss = 0
    code0_loss = 0
    acoustic_loss = 0
    
    device = torch.device("cuda")

    if config.wandb.use_wandb:
        import wandb
        
        wandb.init(project=config.wandb.project_name)
        wandb.config.update(config.model_dump())

    for epoch in range(config.num_epochs):
        pbar = tqdm(train_loader)
        for batch in pbar:
            output = train_step(
                model,
                batch=batch,
                device=device,
                accumulate_step=config.optim.accumulate_steps,
            )
            accumulate_step += 1
            step_loss += output.loss.item()
            code0_loss += output.code0_loss.item()
            acoustic_loss += output.acoustic_loss.item()


            if accumulate_step == config.optim.accumulate_steps:
                global_step += 1
                if config.optim.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.gradient_clip)
                optimizer.step()
                optimizer.zero_grad()
                if config.wandb.use_wandb:
                    wandb.log({
                        "train_loss": step_loss,
                        "code0_loss": code0_loss,
                        "acoustic_loss": acoustic_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "global_step": global_step,
                        "epoch": epoch,
                    })

                lr = scheduler.get_last_lr()[0]
                pbar.set_description(f"epoch {epoch}, step {global_step}, loss {step_loss:.4f}, lr {lr:.2e}")
                accumulate_step = 0
                step_loss = 0
                code0_loss = 0
                acoustic_loss = 0

                # TODO delete this: I have low VRAM so sue me
                torch.cuda.empty_cache()


        # TODO handle scheduling properly
        # scheduler.step()
    return global_step

        

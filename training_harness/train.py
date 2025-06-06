import torch
from einops import rearrange
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .config import TrainingConfig
from typing import NamedTuple, Optional
from tqdm import tqdm
import wandb  # Import wandb
import os # For saving best model
import math # For initializing best_val_loss
import time # For timestamp fallback

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


def train(config: TrainingConfig, train_loader: DataLoader, val_loader: DataLoader, model: torch.nn.Module, shortcut: ShortcutLayer, optimizer: torch.optim.Optimizer):
    global_step = 0
    accumulate_step = 0

    # Logging
    step_loss = 0
    code0_loss = 0
    acoustic_loss = 0

    # Early Stopping & Checkpointing
    # best_val_loss = math.inf
    # epochs_no_improve = 0
    run_id = None # Initialize run_id
    # Ensure checkpoint dir exists before potentially using it
    os.makedirs(config.checkpoint_dir, exist_ok=True) 

    device = torch.device("cuda")

    if config.wandb.use_wandb:
        # Initialize wandb run first
        run = wandb.init(project=config.wandb.project_name)
        run_id = run.id # Get the run ID
        wandb.config.update(config.model_dump())
        # Watch the model gradients and parameters
        wandb.watch(model, log="all", log_freq=config.optim.accumulate_steps * 10)
    else:
        # Use timestamp as a fallback if wandb is not used
        run_id = f"run_{int(time.time())}"


    for epoch in range(config.num_epochs):
        model.train() # Set model to training mode
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        for batch in pbar:
            output = train_step(
                model,
                batch=batch,
                device=device,
                accumulate_step=config.optim.accumulate_steps,
            )
            accumulate_step += 1
            # Use accumulate_step for loss averaging before logging/resetting
            step_loss += output.loss.item() 
            code0_loss += output.code0_loss.item() / config.optim.accumulate_steps
            acoustic_loss += output.acoustic_loss.item() / config.optim.accumulate_steps


            if accumulate_step == config.optim.accumulate_steps:
                global_step += 1
                

                if config.optim.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.gradient_clip)
                optimizer.step()

                optimizer.zero_grad()
                
                lr = optimizer.param_groups[0]['lr'] # Get current LR from optimizer


                if config.wandb.use_wandb:
                    wandb.log({
                        "train_loss_step": step_loss, # Log accumulated loss for the step
                        "train_code0_loss_step": code0_loss,
                        "train_acoustic_loss_step": acoustic_loss,
                        "lr": lr,
                        "global_step": global_step,
                        "epoch": epoch,
                    })

                pbar.set_description(f"epoch {epoch}, step {global_step}, loss {step_loss:.4f}, lr {lr:.2e}")
                accumulate_step = 0
                step_loss = 0
                code0_loss = 0
                acoustic_loss = 0


        # Validation loop
        model.eval() # Set model to evaluation mode
        val_loss_total = 0.0
        val_code0_loss_total = 0.0
        val_acoustic_loss_total = 0.0
        val_steps = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation")
        with torch.no_grad():
            for batch in val_pbar:
                tokens = batch["tokens"].to(device)
                tokens_masks = batch["tokens_masks"].to(device)
                pad_mask = batch["pad_mask"].to(device)
                labels = batch["labels"].to(device)
                acoustic_codes = batch["acoustic_codes"].to(device)

                codebook_logits, compute_amortize_mask = model(tokens=tokens, tokens_mask=tokens_masks, acoustic_codes=acoustic_codes, key_padding_mask=pad_mask)
                loss_components = compute_losses_logits(codebook_logits, labels, compute_amortize_mask)
                
                val_loss_total += loss_components.total_loss.item()
                val_code0_loss_total += loss_components.code0_loss.item()
                val_acoustic_loss_total += loss_components.acoustic_loss.item()
                val_steps += 1
                val_pbar.set_description(f"epoch {epoch}, val_loss {loss_components.total_loss.item():.4f}")


        avg_val_loss = val_loss_total / val_steps if val_steps > 0 else 0
        avg_val_code0_loss = val_code0_loss_total / val_steps if val_steps > 0 else 0
        avg_val_acoustic_loss = val_acoustic_loss_total / val_steps if val_steps > 0 else 0

        print(f"Epoch {epoch}: Average Validation Loss: {avg_val_loss:.4f}")

        if config.wandb.use_wandb:
            wandb.log({
                "val_loss_epoch": avg_val_loss,
                "val_code0_loss_epoch": avg_val_code0_loss,
                "val_acoustic_loss_epoch": avg_val_acoustic_loss,
                "epoch": epoch,
            })


    if config.wandb.use_wandb:
        wandb.finish()

    return global_step

        

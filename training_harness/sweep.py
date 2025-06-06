import torch
from einops import rearrange
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .config import TrainingConfig, OptimParameters, DatasetConfig, WandbConfig # Ensure all necessary configs are imported
from typing import NamedTuple, Optional, Dict, Any, List
from tqdm import tqdm
import wandb
import os
import math
import time
import optuna
import yaml # For loading sweep config
import argparse
import copy # For deepcopying config
from torch.optim.lr_scheduler import LambdaLR # For LR scheduling

from modeling.generator import load_csm_1b
from modeling.models import Model, ModelArgs
from safetensors.torch import load_file
from torch.optim import AdamW
from .data import create_dataloaders # Assuming data.py is in the same directory or accessible

# Determine the script's directory and the base directory for configs
script_dir = os.path.dirname(os.path.abspath(__file__))
default_config_path = os.path.join(script_dir, '..', 'config', 'train_expresso.toml')
default_sweep_config_path = os.path.join(script_dir, '..', 'config', 'sweep.config.yaml')


class TrainStepOutput(NamedTuple):
    loss: float
    code0_loss: float
    acoustic_loss: float

def compute_losses_mse(outputs, targets, tokens_masks):
    semantic_loss = F.mse_loss(outputs, targets, reduction="none")
    acoustic_mask = tokens_masks[:, :, 1:-1]
    semantic_loss[~acoustic_mask] = 0
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
) -> TrainStepOutput:
    """Single training step"""
    tokens = batch["tokens"].to(device)
    tokens_masks = batch["tokens_masks"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    labels = batch["labels"].to(device)
    acoustic_codes = batch["acoustic_codes"].to(device)

    codebook_logits, compute_amortize_mask = model(tokens=tokens, tokens_mask=tokens_masks, acoustic_codes=acoustic_codes, key_padding_mask=pad_mask)
    loss_components = compute_losses_logits(codebook_logits, labels, compute_amortize_mask)
    loss = loss_components.total_loss / accumulate_step

    loss.backward()
    return TrainStepOutput(
        loss=loss,
        code0_loss=loss_components.code0_loss,
        acoustic_loss=loss_components.acoustic_loss,
    )


def train_epoch(
    trial: optuna.Trial, # Optuna trial object
    config: TrainingConfig, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR], # Added scheduler
    epoch: int, # Current epoch number
    global_step_offset: int = 0 # For continuing global_step across epochs if needed
):
    global_step = global_step_offset
    accumulate_step_count = 0 # Renamed from accumulate_step to avoid conflict

    step_loss = 0
    code0_loss = 0
    acoustic_loss = 0
    
    device = torch.device("cuda")
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training Trial {trial.number}")
    for batch_idx, batch in enumerate(pbar):
        output = train_step(
            model,
            batch=batch,
            device=device,
            accumulate_step=config.optim.accumulate_steps,
        )
        accumulate_step_count += 1
        step_loss += output.loss.item() 
        code0_loss += output.code0_loss.item() / config.optim.accumulate_steps
        acoustic_loss += output.acoustic_loss.item() / config.optim.accumulate_steps

        if accumulate_step_count == config.optim.accumulate_steps:
            global_step += 1
            
            if config.optim.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.gradient_clip)
            optimizer.step()
            if scheduler: # Step the scheduler after optimizer
                scheduler.step()
            optimizer.zero_grad()
            
            lr = optimizer.param_groups[0]['lr']

            if config.wandb.use_wandb:
                wandb.log({
                    "train_loss_step": step_loss,
                    "train_code0_loss_step": code0_loss,
                    "train_acoustic_loss_step": acoustic_loss,
                    "lr": lr,
                    "global_step": global_step, # Log combined global_step
                    "epoch": epoch,
                }, step=global_step) # Ensure global_step is used for x-axis

            pbar.set_description(f"Epoch {epoch}, Step {global_step}, Loss {step_loss:.4f}, LR {lr:.2e}")
            accumulate_step_count = 0
            step_loss = 0
            code0_loss = 0
            acoustic_loss = 0
            
            # Optuna Pruning (optional, based on intermediate validation)
            # For simplicity, we'll prune based on end-of-epoch validation loss here.
            # If you want to prune mid-epoch, you'd need a validation step here.

    # Validation loop (at the end of each epoch)
    model.eval()
    val_loss_total = 0.0
    val_code0_loss_total = 0.0
    val_acoustic_loss_total = 0.0
    val_steps = 0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation Trial {trial.number}")
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
            val_pbar.set_description(f"Epoch {epoch}, Val Loss {loss_components.total_loss.item():.4f}")

    avg_val_loss = val_loss_total / val_steps if val_steps > 0 else float('inf')
    avg_val_code0_loss = val_code0_loss_total / val_steps if val_steps > 0 else float('inf')
    avg_val_acoustic_loss = val_acoustic_loss_total / val_steps if val_steps > 0 else float('inf')

    print(f"Epoch {epoch} Trial {trial.number}: Avg Val Loss: {avg_val_loss:.4f}")

    if config.wandb.use_wandb:
        wandb.log({
            "val_loss_epoch": avg_val_loss,
            "val_code0_loss_epoch": avg_val_code0_loss,
            "val_acoustic_loss_epoch": avg_val_acoustic_loss,
            "epoch": epoch,
        }, step=global_step) # Log with global_step

    return avg_val_loss, global_step


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer, 
    lr_decay_type: str, 
    warmup_steps: int, 
    num_training_steps: int
) -> Optional[LambdaLR]:
    
    def lr_lambda_linear(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
        )

    def lr_lambda_cosine(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def lr_lambda_constant(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    if lr_decay_type == "linear":
        return LambdaLR(optimizer, lr_lambda_linear)
    elif lr_decay_type == "cosine":
        return LambdaLR(optimizer, lr_lambda_cosine)
    elif lr_decay_type == "constant":
        return LambdaLR(optimizer, lr_lambda_constant)
    # Add exponential or other schedulers here if needed
    # elif lr_decay_type == "exponential":
    #    # Note: ExponentialLR is usually epoch-stepped. For step-wise, need custom lambda or different scheduler.
    #    # Example: return LambdaLR(optimizer, lambda step: decay_rate ** (step / decay_steps))
    #    pass 
    return None # No scheduler or unknown type

def objective(trial: optuna.Trial, base_config: TrainingConfig, sweep_params_config: Dict[str, Any]):
    # --- Hyperparameter suggestion ---
    cfg = copy.deepcopy(base_config) # Start with a copy of the base config

    cfg.dataset.batch_size = trial.suggest_categorical("batch_size", sweep_params_config["batch_size"]["values"])
    cfg.optim.lr = trial.suggest_float("learning_rate", 
                                     sweep_params_config["learning_rate"]["min"], 
                                     sweep_params_config["learning_rate"]["max"], 
                                     log=sweep_params_config["learning_rate"]["log"])
    beta1 = trial.suggest_float("beta1", 
                              sweep_params_config["beta1"]["min"], 
                              sweep_params_config["beta1"]["max"], 
                              log=sweep_params_config["beta1"].get("log", False))
    beta2 = trial.suggest_float("beta2", 
                              sweep_params_config["beta2"]["min"], 
                              sweep_params_config["beta2"]["max"], 
                              log=sweep_params_config["beta2"].get("log", False))
    cfg.optim.betas = [beta1, beta2]

    cfg.optim.weight_decay = trial.suggest_float("weight_decay", 
                                               sweep_params_config["weight_decay"]["min"], 
                                               sweep_params_config["weight_decay"]["max"], 
                                               log=sweep_params_config["weight_decay"]["log"])
    cfg.optim.gradient_clip = trial.suggest_float("max_grad_norm", 
                                                sweep_params_config["max_grad_norm"]["min"], 
                                                sweep_params_config["max_grad_norm"]["max"],
                                                log=sweep_params_config["max_grad_norm"]["log"]) # Note: schema says log:false
    cfg.optim.accumulate_steps = trial.suggest_int("grad_acc_steps", 
                                                 sweep_params_config["grad_acc_steps"]["min"], 
                                                 sweep_params_config["grad_acc_steps"]["max"])
    cfg.optim.freeze_backbone = trial.suggest_categorical("freeze_backbone", sweep_params_config["freeze_backbone"]["values"])
    
    # LR Decay and Warmup
    lr_decay_type = trial.suggest_categorical("lr_decay", sweep_params_config["lr_decay"]["values"])
    warmup_steps = trial.suggest_int("warmup_steps", 
                                   sweep_params_config["warmup_steps"]["min"], 
                                   sweep_params_config["warmup_steps"]["max"])

    # --- WandB Initialization ---
    run_id = f"optuna_trial_{trial.number}_{int(time.time())}"
    if cfg.wandb.use_wandb:
        # Ensure wandb config uses the project name from sweep_params_config if available
        wandb_project_name = sweep_params_config.get("wandb", {}).get("project_name", cfg.wandb.project_name)
        run = wandb.init(
            project=wandb_project_name,
            config=cfg.model_dump(), # Log the effective config for this trial
            group="optuna_sweep", # Group trials
            name=f"trial-{trial.number}",
            reinit=True # Necessary for multiple trials in one script
        )
        run_id = run.id
    
    # --- Model Loading ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.init_model_path:
        init_model_path_abs = cfg.init_model_path
        if not os.path.isabs(init_model_path_abs):
            # Make relative to the main project dir, not csm_finetune/training_harness
            project_root_dir = os.path.dirname(os.path.dirname(script_dir))
            init_model_path_abs = os.path.join(project_root_dir, init_model_path_abs)

        config_path = os.path.join(init_model_path_abs, "config.json")
        model_file = os.path.join(init_model_path_abs, "model.safetensors")
        
        if not os.path.exists(config_path):
            print(f"Error: Model config.json not found at {config_path}")
            raise FileNotFoundError(f"Model config.json not found at {config_path}")
        if not os.path.exists(model_file):
            print(f"Error: Model model.safetensors not found at {model_file}")
            raise FileNotFoundError(f"Model model.safetensors not found at {model_file}")
            
        with open(config_path, "r") as f:
            model_config_dict = yaml.safe_load(f) # json.load(f)
        model_args = ModelArgs(**model_config_dict)
        model = Model(model_args)
        state_dict = load_file(model_file)
        model.load_state_dict(state_dict, strict=False)
        model.to(device=device, dtype=torch.bfloat16)
    else:
        model = load_csm_1b(str(device), setup_caches=False) # load_csm_1b expects string for device
        model._audio_tokenizer.to("cpu") # Keep audio tokenizer on CPU
        for param in model._audio_tokenizer.parameters():
            param.requires_grad = False
        model = model._model # Get the underlying Transformer model

    # --- Dataloaders ---
    # Ensure dataset_dir is absolute or correctly relative
    dataset_dir_abs = cfg.dataset.dataset_dir
    if not os.path.isabs(dataset_dir_abs):
        project_root_dir = os.path.dirname(os.path.dirname(script_dir)) # csm_finetune dir
        dataset_dir_abs = os.path.join(project_root_dir, dataset_dir_abs)
    
    # Create a temporary dataset config for create_dataloaders
    temp_dataset_cfg = DatasetConfig(
        num_workers=cfg.dataset.num_workers,
        batch_size=cfg.dataset.batch_size, # This is the swept value
        dataset_dir=dataset_dir_abs,
        p_amortize_keep_alive=cfg.dataset.p_amortize_keep_alive
    )
    # Pass the whole cfg to create_dataloaders, it will pick what it needs
    # Or, more cleanly, adjust create_dataloaders to take individual params or a sub-config
    
    # Create a temporary config object for create_dataloaders
    # This assumes create_dataloaders expects a TrainingConfig like object
    # or at least one with a `dataset` attribute that is a DatasetConfig
    temp_loader_config = copy.deepcopy(cfg)
    temp_loader_config.dataset.dataset_dir = dataset_dir_abs # Ensure absolute path
    temp_loader_config.dataset.batch_size = cfg.dataset.batch_size # Use swept batch size
    
    train_loader, val_loader = create_dataloaders(temp_loader_config, None) # Pass MimiModel as None

    # --- Optimizer ---
    if cfg.optim.freeze_backbone:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    else:
        model.p_amortize_keep_alive = cfg.dataset.p_amortize_keep_alive
        model.train()
        
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=cfg.optim.lr, 
                      betas=tuple(cfg.optim.betas), # Ensure betas is a tuple
                      weight_decay=cfg.optim.weight_decay)

    # Calculate total training steps for scheduler
    num_training_steps_per_epoch = math.ceil(len(train_loader) / cfg.optim.accumulate_steps)
    total_num_training_steps = num_training_steps_per_epoch * cfg.num_epochs

    scheduler = get_lr_scheduler(optimizer, lr_decay_type, warmup_steps, total_num_training_steps)

    # --- Training Loop ---
    best_val_loss = float('inf')
    current_global_step = 0
    for epoch in range(cfg.num_epochs):
        avg_val_loss_epoch, current_global_step = train_epoch(
            trial, cfg, train_loader, val_loader, model, optimizer, scheduler, epoch, global_step_offset=current_global_step
        )
        
        # Optuna Pruning (report intermediate value)
        trial.report(avg_val_loss_epoch, epoch)
        if trial.should_prune():
            if cfg.wandb.use_wandb: wandb.finish(exit_code=1) # Mark as pruned
            raise optuna.exceptions.TrialPruned()

        if avg_val_loss_epoch < best_val_loss:
            best_val_loss = avg_val_loss_epoch
            # Optional: Save best model for this trial
            if cfg.checkpoint_dir:
                trial_checkpoint_dir = os.path.join(cfg.checkpoint_dir, f"trial_{trial.number}")
                os.makedirs(trial_checkpoint_dir, exist_ok=True)
                save_path = os.path.join(trial_checkpoint_dir, f"best_model_epoch_{epoch}.pt")
                torch.save(model.state_dict(), save_path)
                if cfg.wandb.use_wandb:
                    wandb.save(save_path) # Save to wandb artifacts

    if cfg.wandb.use_wandb:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.finish()

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Optuna Sweeping Script")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to the base training configuration file (e.g., train_expresso.toml)")
    parser.add_argument("--sweep-config", type=str, default=default_sweep_config_path, help="Path to the Optuna sweep configuration file (e.g., ../config/sweep.config.yaml)")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials to run.")
    parser.add_argument("--study-name", type=str, default="csm_finetune_sweep", help="Name for the Optuna study.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///example.db).")
    
    args = parser.parse_args()

    # --- Load Base Configuration ---
    base_config = TrainingConfig.from_toml(args.config)

    # --- Load Sweep Configuration ---
    with open(args.sweep_config, 'r') as f:
        sweep_config_yaml = yaml.safe_load(f)
    
    # Extract parameters for Optuna's suggest methods
    # This part needs to map sweep_config_yaml to what `trial.suggest_` expects
    # For simplicity, we'll assume sweep_config_yaml directly contains the parameter definitions
    # as used in the objective function.
    # The sweep.config.yaml should define ranges for:
    # batch_size, learning_rate, weight_decay, max_grad_norm, grad_acc_steps

    # --- Optuna Study ---
    # Ensure checkpoint_dir from base_config exists
    if base_config.checkpoint_dir:
        os.makedirs(base_config.checkpoint_dir, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize", # We want to minimize validation loss
        storage=args.storage,
        load_if_exists=True # Resume study if it already exists
    )

    study.optimize(lambda trial: objective(trial, base_config, sweep_config_yaml), 
                   n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best parameters
    best_params_file = os.path.join(base_config.checkpoint_dir, f"{args.study_name}_best_params.yaml")
    with open(best_params_file, 'w') as f:
        yaml.dump(trial.params, f)
    print(f"Best parameters saved to {best_params_file}")


if __name__ == "__main__":
    main() 
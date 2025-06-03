import argparse
import os
from dotenv import load_dotenv
# import pickle # Not used in your original script
import yaml
from pathlib import Path
from tqdm import tqdm
import optuna
import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import wandb
from huggingface_hub import HfApi, create_repo, upload_file # For Hub interaction

# Assuming Model is part of what's imported from utils or CSM_REPO_PATH
from utils import (
    load_model,
    load_tokenizers,
    generate_audio,
    WarmupDecayLR,
    validate,
    load_watermarker,
    MIMI_SAMPLE_RATE,
    Model # Make sure Model class is accessible
)
from dataloaders import create_dataloaders

# Load environment variables from .env file in the script's parent directory
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

if os.getenv("WANDB_API_KEY") is None:
    raise ValueError("WANDB_API_KEY is not set in the .env file")


def parse_args(arg_string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/tokens.hdf5", type=str, help="Path to the pre-tokenized data")
    parser.add_argument("--output_dir", type=Path, default="./exp", help="Path to save the model locally")
    parser.add_argument("--config", type=str, default='./configs/finetune_param_defaults.yaml', help="Path to the finetuning config")
    parser.add_argument("--model_name_or_checkpoint_path", type=str, default="sesame/csm-1b", help="Pretrained model name or path to local checkpoint or huggingface model")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--partial_data_loading", action="store_true", help="Use partial data loading (use for large datasets)")

    parser.add_argument("--wandb_project", type=str, default="csm-finetuning", help="Name of the project")
    parser.add_argument("--wandb_name", type=str, default=None, help="Name of the run")
    parser.add_argument("--wandb_reinit", type=bool, default=True, help="Whether to reinitialize the run")

    parser.add_argument("--log_every", type=int, default=10, help="Log every n steps")
    parser.add_argument("--val_every", type=int, default=100, help="Validate every n steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint locally every n steps")
    parser.add_argument("--gen_every", type=int, default=1000, help="Generate every n steps")
    parser.add_argument(
        "--gen_sentences",
        type=str,
        default="Bird law in this country is not governed by reason.",
        help="Sentence(s) for model to generate. If a path is provided, the model will generate from the sentences in the file.",
    )
    parser.add_argument("--gen_speaker", type=int, default=999, help="Speaker id for model to generate")

    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to train. If not provided, the training will run indefinitely.")

    # --- Hugging Face Hub Saving Arguments ---
    parser.add_argument("--hf_repo_id", type=str, default=None,
                        help="Hugging Face Hub repository ID (e.g., YourUsername/my-model). If not set and --hf_save_to_default_repo is true, defaults to 'LucaZuana/brooke'.")
    parser.add_argument("--hf_save_to_default_repo", action="store_true",
                        help="Enable saving to the default private Hub repo 'LucaZuana/brooke' if --hf_repo_id is not specified.")
    parser.add_argument("--hf_make_public", action="store_true",
                        help="Make the Hugging Face Hub repository public (default is private).")
    # We will push best and final models if Hub saving is active. Periodic pushes can be added if needed.

    args = parser.parse_args(arg_string.split() if arg_string else None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Ensure gen_sentences is Path only if it's a file path string
    if isinstance(args.gen_sentences, str) and args.gen_sentences.endswith(".txt"):
        args.gen_sentences = Path(args.gen_sentences)

    if args.train_from_scratch:
        args.model_name_or_checkpoint_path = None

    # --- Initialize Hub Repo ---
    args.effective_hf_repo_id = args.hf_repo_id
    args.is_hf_repo_private = not args.hf_make_public # Default to private

    if not args.effective_hf_repo_id and args.hf_save_to_default_repo:
        args.effective_hf_repo_id = "LucaZuana/brooke"
        # If --hf_make_public is not set, default repo 'LucaZuana/brooke' will be private.
        print(f"No --hf_repo_id provided, using default Hub repo: {args.effective_hf_repo_id} (Private: {args.is_hf_repo_private})")

    if args.effective_hf_repo_id:
        try:
            print(f"Ensuring Hugging Face Hub repository '{args.effective_hf_repo_id}' (Private: {args.is_hf_repo_private})")
            create_repo(args.effective_hf_repo_id, private=args.is_hf_repo_private, exist_ok=True, repo_type="model")
            print(f"Hub repository '{args.effective_hf_repo_id}' is ready.")
        except Exception as e:
            print(f"Error creating/accessing Hub repo '{args.effective_hf_repo_id}': {e}. Disabling Hub saving.")
            args.effective_hf_repo_id = None # Disable Hub saving on error
    else:
        print("Hugging Face Hub saving is disabled.")
    
    return args


def _save_to_hub_internal(model_to_save: Model, local_checkpoint_path: Path, hub_repo_id: str, commit_message: str, is_private: bool):
    """Internal helper to push to Hub."""
    if not hub_repo_id:
        return
    
    print(f"\nAttempting to save to Hugging Face Hub: {hub_repo_id} (Commit: '{commit_message}')")

    # 1. Push the full .pt checkpoint file (for exact state restoration)
    try:
        remote_pt_filename = local_checkpoint_path.name
        print(f"Uploading full checkpoint '{local_checkpoint_path}' to '{hub_repo_id}/{remote_pt_filename}'...")
        upload_file(
            path_or_fileobj=str(local_checkpoint_path),
            path_in_repo=remote_pt_filename,
            repo_id=hub_repo_id,
            repo_type="model",
            commit_message=f"Upload checkpoint: {commit_message}" # Differentiate commit for .pt
        )
        print(f"Successfully uploaded full checkpoint: {hub_repo_id}/{remote_pt_filename}")
    except Exception as e:
        print(f"Error uploading full .pt checkpoint to Hub: {e}")

    # 2. Push model using model.save_pretrained (for standard Hub model files like config.json, model.safetensors)
    # This makes the model more easily loadable via `Model.from_pretrained(hub_repo_id)`
    temp_save_dir_for_hub = local_checkpoint_path.parent / f"{local_checkpoint_path.stem}_hub_files"
    try:
        print(f"Saving model using save_pretrained to Hub compatible format in {temp_save_dir_for_hub}...")
        model_to_save.save_pretrained(
            save_directory=str(temp_save_dir_for_hub),
            # repo_id=hub_repo_id, # Not needed here if push_to_hub is True, it takes it from context or main repo_id
            # push_to_hub=True,    # We will push the directory content manually for more control if needed
            # private=is_private,
            # commit_message=f"Model files: {commit_message}",
            safe_serialization=True
        )
        print(f"Model saved in Hub format locally at {temp_save_dir_for_hub}. Now pushing to {hub_repo_id}...")
        api = HfApi()
        api.upload_folder(
            folder_path=str(temp_save_dir_for_hub),
            repo_id=hub_repo_id,
            repo_type="model",
            commit_message=f"Model files: {commit_message}" # Separate commit for these files
        )
        print(f"Successfully pushed Hub-formatted model files to {hub_repo_id}")
        # import shutil; shutil.rmtree(temp_save_dir_for_hub) # Optional: clean up temp dir
    except AttributeError:
        print(f"Model object does not have 'save_pretrained' method. Skipping Hub-format model saving.")
    except Exception as e:
        print(f"Error saving/uploading Hub-formatted model files: {e}")


def train(args: argparse.Namespace, config: dict, device: torch.device, trial: optuna.Trial = None):
    """
    trial is only used when we are sweeping hyperparameters.
    """
    assert wandb.run is not None, "Wandb is not initialized"

    eff_batch_size = config["batch_size"] * config["grad_acc_steps"]
    
    model = load_model(model_name_or_checkpoint_path=args.model_name_or_checkpoint_path, device=device, decoder_loss_weight=config["decoder_loss_weight"])
    text_tokenizer, audio_tokenizer = load_tokenizers(device)
    watermarker = load_watermarker(device=device)
    trainloader, valloader = create_dataloaders(
        args.data, 
        config["batch_size"], 
        infinite_train=False, # Set to False as n_epochs is typically used
        load_in_memory=not args.partial_data_loading
    )
    
    # Ensure total_steps is valid for scheduler, even if n_epochs is 0 or None.
    if args.n_epochs and args.n_epochs > 0:
        total_steps = args.n_epochs * len(trainloader)
    else:
        print("Warning: --n_epochs is not set or is <= 0. Training will run for a conceptual single, very long epoch.")
        print("Scheduler will use a large number for total_steps (1,000,000) if lr_decay is not 'constant'.")
        total_steps = 1_000_000 # For scheduler if actual epochs not defined
        args.n_epochs = 1 # Conceptually run 1 very long epoch if total_steps is the driver

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = WarmupDecayLR(optimizer, config["warmup_steps"], total_steps, config["lr_decay"])
    scaler = GradScaler(enabled=args.use_amp)

    current_best_val_loss = float("inf") # Renamed from state["best_val_loss"] for clarity
    
    step = 0
    train_losses = []
    pbar_display_total = total_steps if total_steps < 1_000_000 else None # For TQDM display
    pbar = tqdm(total=pbar_display_total, desc="Training" if trial is None else f"Trial {trial.number}")
    
    model.train()
    
    for epoch in range(args.n_epochs): # Loop for the specified number of epochs
        if pbar_display_total and step >= pbar_display_total: # Exit if fixed total_steps reached
            print(f"Reached target total_steps ({pbar_display_total}). Stopping training.")
            break

        for tokens, tokens_mask in trainloader:
            if pbar_display_total and step >= pbar_display_total: break # Inner loop check

            tokens, tokens_mask = tokens.to(device), tokens_mask.to(device)
                
            with autocast(device_type=str(device), enabled=args.use_amp):
                loss_val = model(tokens, tokens_mask) # Get the single loss tensor
                loss_for_backward = loss_val / config["grad_acc_steps"]
            
            scaler.scale(loss_for_backward).backward()
            
            if (step + 1) % config["grad_acc_steps"] == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step() # Step the scheduler after optimizer step
            
            # Log the actual loss for this step (not averaged over grad_acc_steps for this item)
            train_losses.append(loss_val.item()) 
            
            if args.log_every > 0 and step % args.log_every == 0:
                avg_loss = sum(train_losses) / len(train_losses) if train_losses else 0
                wandb.log({
                        "train_loss_avg": avg_loss, "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }, step=step)
                train_losses = [] # Reset for next logging interval

            # --- Local Checkpoint Saving ---
            # Save if save_every is met OR it's the last step of a finite training run
            is_last_step_of_finite_training = pbar_display_total and (step == pbar_display_total - 1)
            if args.save_every > 0 and step > 0 and (step % args.save_every == 0 or is_last_step_of_finite_training):
                local_checkpoint_path = args.output_dir / f"model_step_{step}.pt"
                # Prepare comprehensive state for saving
                state_to_save = {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "effective_batch_size": eff_batch_size, "config": config, "args": vars(args),
                    "best_val_loss": current_best_val_loss, "step": step, "epoch": epoch
                }
                torch.save(state_to_save, local_checkpoint_path)
                print(f"\nSaved local checkpoint: {local_checkpoint_path}")
                # Note: Periodic Hub pushes for non-best/final checkpoints are omitted for simplicity as requested.

            # --- Validation ---
            if args.val_every > 0 and step > 0 and (step % args.val_every == 0 or is_last_step_of_finite_training):
                val_loss = validate(model, valloader, device, args.use_amp)
                wandb.log({"val_loss": val_loss}, step=step)

                if val_loss < current_best_val_loss:
                    current_best_val_loss = val_loss
                    local_best_val_path = args.output_dir / "model_bestval.pt" # Always overwrite best
                    state_to_save_best = { # Create specific state for best model
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "effective_batch_size": eff_batch_size, "config": config, "args": vars(args),
                        "best_val_loss": current_best_val_loss, "step": step, "epoch": epoch
                    }
                    torch.save(state_to_save_best, local_best_val_path)
                    print(f"\nSaved new best validation model locally: {local_best_val_path} (Val Loss: {val_loss:.4f})")
                    wandb.save(str(local_best_val_path)) # Log best model to W&B artifacts

                    # --- Save Best Model to Hugging Face Hub ---
                    if args.effective_hf_repo_id: # Check if Hub saving is active
                        _save_to_hub_internal(model, local_best_val_path, args.effective_hf_repo_id,
                                              f"Best validation model (step {step}, loss {val_loss:.4f})",
                                              args.is_hf_repo_private)
                
                if trial is not None: # Optuna sweep handling
                    trial.report(val_loss, step)
                    if trial.should_prune():
                        wandb.finish(); pbar.close(); raise optuna.exceptions.TrialPruned()
                
                model.train() # Ensure model is back in train mode
                pbar.set_postfix({"train_loss": f"{loss_val.item():.4f}", "val_loss": f"{val_loss:.4f}"})
            else:
                pbar.set_postfix({"train_loss": f"{loss_val.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}", "epoch": epoch})
            
            # --- Generation ---
            if args.gen_every > 0 and step > 0 and (step % args.gen_every == 0) and not (args.train_from_scratch and step == 0):
                gen_sents = []
                if isinstance(args.gen_sentences, str): gen_sents.append(args.gen_sentences)
                elif isinstance(args.gen_sentences, Path):
                    try:
                        with open(args.gen_sentences, "r") as f: gen_sents = [ln.strip() for ln in f if ln.strip()]
                    except FileNotFoundError: print(f"Warning: Gen sentences file not found: {args.gen_sentences}")
                
                for i, sentence_txt in enumerate(gen_sents):
                    if not sentence_txt: continue
                    audio_output = generate_audio(model, audio_tokenizer, text_tokenizer, watermarker,
                                                  sentence_txt, args.gen_speaker, device, use_amp=args.use_amp)
                    wandb.log({f"audio_gen_{i}_{sentence_txt[:20]}": wandb.Audio(audio_output, sample_rate=MIMI_SAMPLE_RATE)}, step=step)
                model.train() # Ensure model is back in train mode
            
            pbar.update(1)
            step += 1
        # End of inner loop (dataloader)
    # End of outer loop (epochs)
    
    pbar.close()

    # --- Save Final Model Locally ---
    final_model_local_path = args.output_dir / "model_final.pt"
    final_state_to_save = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "effective_batch_size": eff_batch_size, "config": config, "args": vars(args),
        "best_val_loss": current_best_val_loss, "step": step -1 , # step is one ahead due to pbar update
        "epoch": epoch # Last completed epoch
    }
    torch.save(final_state_to_save, final_model_local_path)
    print(f"\nSaved final model locally: {final_model_local_path}")

    # --- Save Final Model to Hugging Face Hub ---
    if args.effective_hf_repo_id: # Check if Hub saving is active
        _save_to_hub_internal(model, final_model_local_path, args.effective_hf_repo_id,
                              f"Final model (epoch {epoch+1}, step {step})",
                              args.is_hf_repo_private)

    return current_best_val_loss


if __name__ == "__main__":
    args = parse_args() # This will handle Hub repo initialization and defaults
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"train_bs-{config['batch_size']}x{config['grad_acc_steps']}",
        notes=f"Config: {args.config}, Model: {args.model_name_or_checkpoint_path}",
        config={**config, **vars(args)}, # Log combined config
        reinit=args.wandb_reinit,
        dir=str(args.output_dir / "wandb"), # Ensure path is string
    )

    final_val_loss_result = train(args, config, device)
    print(f"\nTraining finished. Best validation loss: {final_val_loss_result:.4f}")

    wandb.finish()
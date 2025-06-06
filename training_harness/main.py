import argparse
from training_harness.config import TrainingConfig
from training_harness.train import train
from training_harness.data import create_dataloaders
from modeling.generator import load_csm_1b
from modeling.models import Model, ModelArgs
from modeling.shortcut_layer import ShortcutLayer
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
import torch
import json
import os

# Determine the script's directory and the base directory for configs
script_dir = os.path.dirname(os.path.abspath(__file__))
# config is expected in ../config relative to this script
default_config_path = os.path.join(script_dir, '..', 'config', 'train_expresso.toml')


parser = argparse.ArgumentParser(description="Training script")
# Use the calculated absolute path as the default
parser.add_argument("--config", type=str, default=default_config_path, help="Path to the configuration file")


def main():
    args = parser.parse_args()
    
    # Use the resolved config path directly
    config_path_to_load = args.config
    # Check if the provided path exists, otherwise try resolving relative to CWD if it looks relative
    if not os.path.exists(config_path_to_load) and not os.path.isabs(config_path_to_load):
        print(f"Warning: Config path '{config_path_to_load}' not found directly. Trying relative to current working directory.")
        config_path_to_load = os.path.join(os.getcwd(), config_path_to_load)
        if not os.path.exists(config_path_to_load):
             # Fallback or error if still not found - using original provided path for error message
             print(f"Error: Config file '{args.config}' not found, neither directly nor relative to CWD.")
             # Optionally raise an error here or exit
             # raise FileNotFoundError(f"Config file '{args.config}' not found.")
             # For now, let TrainingConfig handle the potential FileNotFoundError downstream
             config_path_to_load = args.config # Revert to original path for error reporting in from_toml

    config = TrainingConfig.from_toml(config_path_to_load)

    # TODO make this configurable
    # shortcut = ShortcutLayer(1024, 256)

    # load model
    if config.init_model_path is not None:
        # Convert relative path to absolute path if necessary
        init_model_path = config.init_model_path
        if not os.path.isabs(init_model_path):
            # If path is relative, make it relative to the script directory
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            init_model_path = os.path.join(script_dir, init_model_path)
        
        config_path = os.path.join(init_model_path, "config.json")
        print(f"Loading model config from: {config_path}")
        
        with open(config_path, "r") as f:
            model_config = json.load(f)
        model = Model(ModelArgs(**model_config))
        
        model_file = os.path.join(init_model_path, "model.safetensors")
        print(f"Loading model weights from: {model_file}")
        
        state_dict = load_file(model_file)
        model.load_state_dict(state_dict, strict=False)
        model.to(device="cuda", dtype=torch.bfloat16)
        # model.decoder = torch.compile(model.decoder)
    else:
        model = load_csm_1b("cuda", setup_caches=False)
        model._audio_tokenizer.to("cpu")
        for param in model._audio_tokenizer.parameters():
            param.requires_grad = False
        model = model._model

    # TODO use an actually good dataset if this doesn't generalize
    train_loader, val_loader = create_dataloaders(config, None)
    if config.optim.freeze_backbone:
        # Freeze CSM and Mimi
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    else:
        model.p_amortize_keep_alive = config.dataset.p_amortize_keep_alive
        model.train()
        

    # Load optimizer
    optimizer = AdamW(model.parameters(), lr=config.optim.lr, betas=config.optim.betas, weight_decay=config.optim.weight_decay)

    # Train
    train(config, train_loader, val_loader, model, None, optimizer)


    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{config.checkpoint_dir}/brooke_model4.pt")




if __name__ == "__main__":
    main()

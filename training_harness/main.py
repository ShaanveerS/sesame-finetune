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


parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--config", type=str, default="../config/train_expresso.toml", help="Path to the configuration file")


def main():
    args = parser.parse_args()
    config = TrainingConfig.from_toml(args.config)

    # TODO make this configurable
    # shortcut = ShortcutLayer(1024, 256)

    # load model
    if config.init_model_path is not None:
        with open(f"{config.init_model_path}/config.json", "r") as f:
            model_config = json.load(f)
        model = Model(ModelArgs(**model_config))
        state_dict = load_file(f"{config.init_model_path}/model.safetensors")
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
    optimizer = AdamW(model.parameters(), lr=config.optim.lr, betas=config.optim.betas)

    # Load scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: config.optim.lr ** (epoch / config.num_epochs)
    )

    # Train
    train(config, train_loader, val_loader, model, None, optimizer, scheduler)

    # Save final model
    # TODO: proper checkpointing
    torch.save(model.state_dict(), f"{config.checkpoint_dir}/final_model.pt")




if __name__ == "__main__":
    main()

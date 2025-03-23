from training_harness.config import TrainingConfig
from training_harness.train import train
from training_harness.data import create_dataloaders
from modeling.generator import load_csm_1b
from modeling.shortcut_layer import ShortcutLayer
from torch.optim import AdamW
import torch


def main():
    config = TrainingConfig.from_toml("../config/train_shortcut.toml")

    # TODO make this configurable
    shortcut = ShortcutLayer(1024, 256)

    # load model
    model = load_csm_1b("cuda", setup_caches=False)
    # TODO use an actually good dataset if this doesn't generalize
    train_loader, val_loader = create_dataloaders(config, model._audio_tokenizer)

    # move mimi model back to CPU
    model._audio_tokenizer.to("cpu")


    for param in model._audio_tokenizer.parameters():
        param.requires_grad = False
    if config.optim.freeze_backbone:
        # Freeze CSM and Mimi
        model._model.eval()
        for param in model._model.parameters():
            param.requires_grad = False

    else:
        model._model.p_amortize_keep_alive = config.dataset.p_amortize_keep_alive
        model._model.train()
    
        

    # Load optimizer
    optimizer = AdamW(model._model.parameters(), lr=config.optim.lr, betas=config.optim.betas)

    # Load scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: config.optim.lr ** (epoch / config.num_epochs)
    )

    # Train
    train(config, train_loader, val_loader, model._model, shortcut, optimizer, scheduler)



if __name__ == "__main__":
    main()

from train.config import TrainingConfig
from train.train import step
from train.data import load_mimi_ds
from modeling.generator import load_csm_1b


def main():
    config = TrainingConfig.from_toml("../config/train_shortcut.toml")

    # TODO use an actually good dataset if this doesn't generalize
    train_ds, val_ds = load_mimi_ds(config)

    # load model
    model = load_csm_1b("cuda")

    # Freeze CSM and Mimi
    for param in model._model.parameters():
        param.requires_grad = False

    for param in model._audio_tokenizer.parameters():
        param.requires_grad = False


if __name__ == "__main__":
    main()

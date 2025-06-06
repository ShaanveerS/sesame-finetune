from pydantic import BaseModel
import tomllib
from typing import List, Optional


class WandbConfig(BaseModel):
    project_name: str
    use_wandb: bool = False


class OptimParameters(BaseModel):
    betas: List[float]
    lr: float = 3e-4
    gradient_clip: float = 0.0
    freeze_backbone: bool = False
    accumulate_steps: int = 1
    weight_decay: float = 0.01 # Default value, adjust in TOML config

class DatasetConfig(BaseModel):
    num_workers: int = 12
    batch_size: int = 8
    dataset_dir: str
    p_amortize_keep_alive: float = 0.0625

class TrainingConfig(BaseModel):
    """
    yes in theory i should use hydra, but whatever
    """

    checkpoint_dir: str
    init_model_path: Optional[str] = None
    """
    Defaults to official model
    """
    save_every_n_steps: int = 5000
    val_size: int = 5000
    accumulate_steps: int = 8
    wandb: WandbConfig
    max_sequence_length: int = 768
    optim: OptimParameters
    num_epochs: int = 1
    dataset: DatasetConfig

    @classmethod
    def from_toml(cls, path: str) -> "TrainingConfig":
        with open(path, "rb") as f:  # tomllib requires binary mode
            data = tomllib.load(f)
        return cls(**data)

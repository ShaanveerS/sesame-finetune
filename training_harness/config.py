from pydantic import BaseModel
import tomllib
from typing import List


class WandbConfig(BaseModel):
    project_name: str
    use_wandb: bool = False


class OptimParameters(BaseModel):
    betas: List[float]
    lr: float = 3e-4
    gradient_clip: float = 0.0

class DatasetConfig(BaseModel):
    num_workers: int = 12
    batch_size: int = 8
    dataset_dir: str

class TrainingConfig(BaseModel):
    """
    yes in theory i should use hydra, but whatever
    """

    checkpoint_dir: str
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

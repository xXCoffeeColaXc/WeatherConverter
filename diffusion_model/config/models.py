from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class TransformConfig(BaseModel):
    resize_resolution: List[int] = Field(..., min_items=2, max_items=2)
    target_resolution: List[int] = Field(..., min_items=2, max_items=2)
    mean: List[float] = Field(..., min_items=3, max_items=3)
    std: List[float] = Field(..., min_items=3, max_items=3)
    horizontal_flip: float = Field(..., ge=0.0, le=1.0)


class DataConfig(BaseModel):
    root_dir: str
    labels: str
    images: str
    train_split: str
    val_split: str
    weather: List[str]
    transform: TransformConfig


class ModelConfig(BaseModel):
    name: str
    im_channels: int
    im_size: int
    down_channels: List[int]
    mid_channels: List[int]
    down_sample: List[bool]
    time_emb_dim: int
    num_down_layers: int
    num_mid_layers: int
    num_up_layers: int
    num_heads: int


class FolderConfig(BaseModel):
    output: str
    weights: str
    logs: str
    checkpoints: str
    samples: str


class TrainingConfig(BaseModel):
    device: str = 'cuda'
    random_seed: int
    epochs: int
    batch_size: int  # Must be greater than 1
    num_workers: int
    lr: float
    log_interval: int
    save_interval: int
    resume_training: bool
    resume_checkpoint: Optional[str] = None
    task_name: str
    sample_size: int
    num_grid_rows: int
    ckpt_name: str


class Config(BaseModel):
    training: TrainingConfig
    data: DataConfig
    model: ModelConfig
    folders: FolderConfig


# Example usage:
if __name__ == '__main__':
    import yaml

    def load_config(config_path: str) -> Config:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return Config(**config_data)

    def dummy_function(ignore_index=None, reduction=None):
        print(f"ignore_index: {ignore_index}")
        print(f"reduction: {reduction}")

    config = load_config('diffusion_model/config/config.yaml')
    print(config.training.epochs)
    print(config.data.transform.target_resolution)

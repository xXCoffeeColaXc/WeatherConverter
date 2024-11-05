from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class DataConfig(BaseModel):
    root_dir: str
    labels: str
    images: str
    weather: List[str]
    image_size: List[int]


class DiffusionConfig(BaseModel):
    num_timesteps: int
    beta_start: float
    beta_end: float


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
    attn_resolutions: List[int]


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
    sample_interval: int
    resume_training: bool
    resume_checkpoint: Optional[str] = None
    sample_size: int
    num_grid_rows: int


class Config(BaseModel):
    training: TrainingConfig
    diffusion: DiffusionConfig
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

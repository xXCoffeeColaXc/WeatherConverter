from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SchedulerConfig(BaseModel):
    type: str
    params: Dict[str, Any]


class LossFunctionConfig(BaseModel):
    type: str
    params: Dict[str, Any]


class TransformConfig(BaseModel):
    target_resolution: List[int] = Field(..., min_items=2, max_items=2)
    mean: List[float] = Field(..., min_items=3, max_items=3)
    std: List[float] = Field(..., min_items=3, max_items=3)
    horizontal_flip: float = Field(..., ge=0.0, le=1.0)
    jitter: Optional[Dict[str, float]]
    random_noise: Optional[Dict[str, Any]] = Field(default=None)
    class_wise_masking: Optional[Dict[str, Any]] = Field(default=None)


class DataConfig(BaseModel):
    root_dir: str
    labels: str
    images: str
    train_split: str
    val_split: str
    weather: List[str]
    transform: TransformConfig


class OptimizerConfig(BaseModel):
    type: str
    params: Dict[str, float]
    layerwise_lr: Dict[str, float]


class ModelConfig(BaseModel):
    path: str
    name: str
    num_classes: int
    output_stride: int
    bn_momentum: float


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
    log_interval: int
    save_interval: int
    resume_training: bool
    resume_checkpoint: Optional[str] = None
    loss_function: LossFunctionConfig
    scheduler: SchedulerConfig


class Config(BaseModel):
    training: TrainingConfig
    data: DataConfig
    optimizer: OptimizerConfig
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

    config = load_config('seg_model/config/config.yaml')
    print(config.training.epochs)
    print(config.data.transform.target_resolution)

    if config.training.loss_function.type == 'CrossEntropyLoss':
        dummy_function(**config.training.loss_function.params)

import torch
import numpy as np
from PIL import Image
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import seg_model.network as network
from seg_model.datasets.acdc import ACDCDataset
from seg_model.utils.utils import Denormalize

from seg_model.metrics.stream_metrics import StreamSegMetrics
from seg_model.config.models import Config, ModelConfig
from seg_model.utils.ext_transforms import *
from seg_model.utils.utils import print_class_iou
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEG_MEAN, SEG_STD = [0.4865, 0.4998, 0.4323], [0.2326, 0.2276, 0.2659]


def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)


# Load the model
def load_model(model_path: Path, model_config: ModelConfig) -> torch.nn.Module:
    model: torch.nn.Module = network.modeling.__dict__[
        model_config.name](num_classes=model_config.num_classes, output_stride=model_config.output_stride)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def validate(model: torch.nn.Module, dataloader: DataLoader) -> dict:
    model.eval()
    metrics.reset()

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(dataloader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

        score = metrics.get_results()
    return score


if __name__ == '__main__':

    config = load_config('seg_model/config/config.yaml')

    model_path = Path(config.folders.checkpoints) / 'no_augmented/deeplabv3plus_resnet101_epoch_40.pth'
    model = load_model(model_path, config.model)

    val_transform = ExtCompose(
        [
            ExtResize(size=config.data.transform.resize_resolution),
            ExtCenterCrop(size=config.data.transform.target_resolution),
            ExtToTensor(),
            ExtNormalize(mean=SEG_MEAN, std=SEG_STD),
        ]
    )

    dataset = ACDCDataset(
        root_dir=config.data.root_dir,
        split=config.data.val_split,
        weather=config.data.weather,
        transform=val_transform
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the dataset path.")

    # Create the DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )

    metrics = StreamSegMetrics(config.model.num_classes)

    val_score = validate(model=model, dataloader=loader)
    print(f"[Val] Overall Acc - Score: {val_score['Overall Acc']:.4f}")
    print(f"[Val] Mean IoU - Score: {val_score['Mean IoU']:.4f}")
    for class_id, iou_value in val_score['Class IoU'].items():
        print(f"    Class {ACDCDataset.class_id_to_name[class_id]}: {iou_value}")

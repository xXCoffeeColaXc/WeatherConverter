import network
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import utils
from datasets.acdc import ACDCDataset
from torch.utils.data import DataLoader
import random
import yaml
from typing import Union
from utils.ext_transforms import *
from utils.utils import create_run, denormalize, print_class_iou
from typing import Optional, Union, List, Tuple, Dict, Any, Callable
from tqdm import tqdm
from metrics.stream_metrics import StreamSegMetrics
from config.models import Config
import wandb


def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)


config = load_config('seg_model/config/config.yaml')
run_id = create_run()

# Setup random seed
random_seed = config.training.random_seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'
                     ) if config.training.device == 'auto' else torch.device(config.training.device)
print("Device: %s" % device)
metrics = StreamSegMetrics(config.model.num_classes)


def setup_logger():
    wandb.init(
        project='weather-converter-seg',
        config={
            "image_size": config.data.transform.target_resolution,
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "lr": config.optimizer.params['lr'],
            "sceduler": config.training.scheduler.type,
            "loss_function": config.training.loss_function.type,
            "optimizer": config.optimizer.type,
            "random_seed": config.training.random_seed,
        }
    )
    # Ensure DEVICE is tracked in WandB
    wandb.config.update({"device": device})


def load_model(weight_path: str) -> torch.nn.Module:
    model = network.modeling.__dict__[config.model.name
                                     ](num_classes=config.model.num_classes, output_stride=config.model.output_stride)
    model.load_state_dict(torch.load(weight_path)['model_state'])
    model.to(device)

    utils.set_bn_momentum(model.backbone, momentum=config.model.bn_momentum)

    return model


def reload_modules(weight_path: str, optimizer: torch.optim.Optimizer, scheduler: utils.PolyLR) -> torch.nn.Module:
    model = network.modeling.__dict__[config.model.name
                                     ](num_classes=config.model.num_classes, output_stride=config.model.output_stride)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    utils.set_bn_momentum(model.backbone, momentum=config.model.bn_momentum)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Model reloaded from {weight_path} at epoch {start_epoch}")

    return model, optimizer, scheduler, start_epoch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: utils.PolyLR,
    epoch: int,
    avg_epoch_loss: float
):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_epoch_loss,
    }
    save_path = f"seg_model/outputs/checkpoints/{run_id}/{config.model.name}"
    torch.save(checkpoint, f"{save_path}_epoch_{epoch}.pth")
    print(f"Checkpoint saved at epoch {epoch}")


def get_dataloader(
    root_dir: str = '../data',
    split: str = 'train',
    weather: Union[str, list[str]] = 'all',
    transform: ExtCompose = None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    # Create the dataset
    dataset = ACDCDataset(root_dir=root_dir, split=split, weather=weather, transform=transform)

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the dataset path.")

    # Create the DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader


def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None
) -> None:

    best_score = 0.0
    start_epoch = 1
    epochs = config.training.epochs

    if config.training.resume_training and config.training.resume_checkpoint is not None:
        model, optimizer, scheduler, start_epoch = reload_modules(config.training.resume_checkpoint, optimizer, scheduler)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        interval_loss = 0.0
        total_loss = 0.0
        cur_itrs = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            total_loss += np_loss

            if cur_itrs % config.training.log_interval == 0:
                avg_interval_loss = interval_loss / config.training.log_interval
                current_lr = scheduler.get_last_lr()[0]  # Optionally, log the current learning rate
                print(
                    f"Epoch [{epoch}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {avg_interval_loss:.4f}, Current LR: {current_lr:.8f}"
                )
                wandb.log({"train_loss": avg_interval_loss})

                interval_loss = 0.0

            # Step the scheduler after each batch
            scheduler.step()

        # Logging at the end of each epoch
        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

        # Save intermediate weights
        if epoch % config.training.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, avg_epoch_loss)

        # Perform validation
        if val_loader is not None:
            val_score = validate(model=model, dataloader=val_loader)
            print(f"[Val] Overall Acc at Epoch [{epoch}/{epochs}] - Score: {val_score['Overall Acc']:.4f}")
            print(f"[Val] Mean IoU at Epoch [{epoch}/{epochs}] - Score: {val_score['Mean IoU']:.4f}")
            print_class_iou(epoch, epochs, val_score)
            wandb.log({"val_mean_iou": val_score['Mean IoU'], "val_overall_acc": val_score['Overall Acc']})

            # Save the best model
            if val_score['Mean IoU'] > best_score:
                best_score = val_score['Mean IoU']
                print(f"New best model found with score: {best_score:.4f}")
                save_checkpoint(model, optimizer, scheduler, epoch, avg_epoch_loss)


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
    transform_config = config.data.transform
    model = load_model(config.model.path)
    setup_logger()

    # Define image transforms
    transform = ExtCompose(
        [
            ExtRandomHorizontalFlip(p=transform_config.horizontal_flip),
            ExtRandomCrop(size=transform_config.target_resolution),
            ExtColorJitter(**transform_config.jitter),
            AddGaussianNoise(**transform_config.random_noise),
            ClassWiseMasking(**transform_config.class_wise_masking),
            ExtToTensor(),
            ExtNormalize(mean=transform_config.mean, std=transform_config.std),
        ]
    )

    val_transform = ExtCompose([
        ExtToTensor(),
        ExtNormalize(mean=transform_config.mean, std=transform_config.std),
    ])

    train_loader = get_dataloader(
        root_dir=config.data.root_dir,
        split=config.data.train_split,
        weather=config.data.weather,
        transform=transform,
        shuffle=True,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )
    val_loader = get_dataloader(
        root_dir=config.data.root_dir,
        split=config.data.val_split,
        weather=config.data.weather,
        transform=val_transform,
        shuffle=False,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )

    total_itrs = len(train_loader) * config.training.epochs

    # Set up optimizer
    if config.optimizer.type == 'SGD':
        optimizer = torch.optim.SGD(
            params=[
                {
                    'params': model.backbone.parameters(), 'lr': 0.1 * config.optimizer.params['lr']
                },
                {
                    'params': model.classifier.parameters(), 'lr': config.optimizer.params['lr']
                },
            ],
            **config.optimizer.params
        )
    else:
        raise ValueError("Optimizer not supported")

    if config.training.scheduler.type == 'PolyLR':
        scheduler = utils.PolyLR(optimizer, total_itrs, **config.training.scheduler.params)
    elif config.training.scheduler.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config.training.scheduler.params)
    else:
        raise ValueError("Scheduler not supported")

    # Define the loss function
    if config.training.loss_function.type == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss(**config.training.loss_function.params)
    elif config.training.loss_function.type == 'FocalLoss':
        criterion = utils.FocalLoss(**config.training.loss_function.params)

    train_loop(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader
    )

    wandb.finish()

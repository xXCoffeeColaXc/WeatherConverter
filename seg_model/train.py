import network
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import utils
from datasets.acdc import ACDCDataset
from torch.utils.data import DataLoader
import random
from typing import Union
from utils.ext_transforms import *
from utils.utils import create_run, denormalize, print_class_iou
from typing import Optional, Union, List, Tuple, Dict, Any, Callable
from tqdm import tqdm
from metrics.stream_metrics import StreamSegMetrics

MODEL_NAME = 'deeplabv3plus_resnet101'
NUM_CLASSES = 19
OUTPUT_STRIDE = 16
#TARGET_RESOLUTION = (540 // 2, 960 // 2)
TARGET_RESOLUTION = (768 // 32, 768 // 32)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s" % device)

# Setup random seed
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

run_id = create_run()

# Define image transforms
transform = ExtCompose(
    [
        ExtRandomHorizontalFlip(p=0.5),
        ExtRandomCrop(size=TARGET_RESOLUTION),
        ExtColorJitter(brightness=0.5, contrast=0.3, saturation=0.3),
        AddGaussianNoise(mean=0.25, std_range=(0.0, 0.1)),
        ClassWiseMasking(p=0.1, num_classes_to_keep=1),
        ExtToTensor(),
        ExtNormalize(mean=MEAN, std=STD),
    ]
)

val_transform = ExtCompose([
    ExtToTensor(),
    ExtNormalize(mean=MEAN, std=STD),
])

metrics = StreamSegMetrics(NUM_CLASSES)


def load_model(weight_path: str) -> torch.nn.Module:
    model = network.modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=OUTPUT_STRIDE)
    model.load_state_dict(torch.load(weight_path)['model_state'])
    model.to(device)

    utils.set_bn_momentum(model.backbone, momentum=0.01)

    return model


def reload_modules(weight_path: str, optimizer: torch.optim.Optimizer, scheduler: utils.PolyLR) -> torch.nn.Module:
    model = network.modeling.__dict__[MODEL_NAME](num_classes=NUM_CLASSES, output_stride=OUTPUT_STRIDE)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

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
    save_path = f"seg_model/outputs/checkpoints/{run_id}/{MODEL_NAME}"
    torch.save(checkpoint, f"{save_path}_epoch_{epoch}.pth")
    print(f"Checkpoint saved at epoch {epoch}")


def get_dataloader(
    root_dir: str = '../data',
    split: str = 'train',
    weather: Union[str, list[str]] = 'all',
    transform: ExtCompose = transform,
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
    scheduler: utils.PolyLR,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    device: torch.device = device,
    epochs: int = 5,
    log_interval: int = 10,  # iterations
    save_interval: int = 5,  # epochs
    resume_training: bool = False,
    weight_path: str = None
) -> None:

    best_score = 0.0
    start_epoch = 1
    if resume_training and weight_path is not None:
        model, optimizer, scheduler, start_epoch = reload_modules(weight_path, optimizer, scheduler)

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

            if cur_itrs % log_interval == 0:
                avg_interval_loss = interval_loss / log_interval
                current_lr = scheduler.get_last_lr()[0]  # Optionally, log the current learning rate
                print(
                    f"Epoch [{epoch}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {avg_interval_loss:.4f}, Current LR: {current_lr:.8f}"
                )
                interval_loss = 0.0

            # Step the scheduler after each batch
            scheduler.step()

        # Logging at the end of each epoch
        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

        # Save intermediate weights
        if epoch % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, avg_epoch_loss)

        # Perform validation
        if val_loader is not None:
            val_score = validate(model=model, dataloader=val_loader, device=device)
            print(f"[Val] Overall Acc at Epoch [{epoch}/{epochs}] - Score: {val_score['Overall Acc']:.4f}")
            print(f"[Val] Mean IoU at Epoch [{epoch}/{epochs}] - Score: {val_score['Mean IoU']:.4f}")
            print_class_iou(epoch, epochs, val_score)

            # Save the best model
            if val_score['Mean IoU'] > best_score:
                best_score = val_score['Mean IoU']
                print(f"New best model found with score: {best_score:.4f}")
                save_checkpoint(model, optimizer, scheduler, epoch, avg_epoch_loss)


def validate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
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

    # TODO: argparser
    # TODO: config manager
    lr = 0.0001
    weight_decay = 1e-4
    epochs = 5
    batch_size = 2
    num_workers = 1
    log_interval = 10  # iterations
    save_interval = 5  # epochs

    model = load_model('seg_model/weights/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar')

    train_loader = get_dataloader(
        root_dir='data',
        split='train',
        weather=['fog'],
        transform=transform,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    val_loader = get_dataloader(
        root_dir='data',
        split='val',
        weather=['fog'],
        transform=val_transform,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers
    )

    total_itrs = len(train_loader) * epochs

    # Set up optimizer
    optimizer = torch.optim.SGD(
        params=[
            {
                'params': model.backbone.parameters(), 'lr': 0.1 * lr
            },
            {
                'params': model.classifier.parameters(), 'lr': lr
            },
        ],
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    #criterion = utils.FocalLoss(ignore_index=255, size_average=True)

    train_loop(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        log_interval=log_interval,
        save_interval=save_interval
    )

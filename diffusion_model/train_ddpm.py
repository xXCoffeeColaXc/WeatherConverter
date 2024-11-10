import yaml
import os
import numpy as np
import wandb
from pathlib import Path
from tqdm import tqdm
import random
import torch
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

from diffusion_model.dataloader import ACDCDataset, get_loader
from diffusion_model.models.unet_base import Unet
from diffusion_model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from diffusion_model.config.models import Config
from diffusion_model.utils import create_run
from diffusion_model.sample_ddpm import sample


def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)


config = load_config('diffusion_model_v2/config/config.yaml')
run_id = create_run()

# Setup random seed
random_seed = config.training.random_seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'
                     ) if config.training.device == 'auto' else torch.device(config.training.device)
print("Device: %s" % device)


def setup_logger():
    wandb.init(
        project='weather-converter-diffusion',
        config={
            "image_size": config.data.image_size,
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "lr": config.training.lr,
            "random_seed": config.training.random_seed,
        }
    )
    # Ensure DEVICE is tracked in WandB
    wandb.config.update({"device": device})


def save_checkpoint(epoch: int, model: Unet, opt: Adam):
    save_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'epoch': epoch}
    save_path = os.path.join(config.folders.checkpoints, run_id, f'{epoch}-checkpoint.ckpt')
    torch.save(save_dict, save_path)
    print('Saved checkpoints into {}...'.format(save_path))


def load_checkpoint(model: Unet, opt: Adam, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, opt, epoch


def train(
    dataloader: DataLoader, model: Unet, optimizer: Adam, criterion: torch.nn.MSELoss, scheduler: LinearNoiseScheduler
):
    start_epoch = 1
    total_loss = 0.0
    interval_loss = 0.0
    num_batches = len(dataloader)
    num_epochs = config.training.epochs
    log_interval = config.training.log_interval

    if config.training.resume_training:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, config.training.resume_checkpoint)
        num_epochs += start_epoch
        print(f"Resuming training from epoch {start_epoch}")

    # Run training
    for epoch_idx in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0
        interval_loss = 0.0

        pbar = tqdm(dataloader, leave=True)

        for batch_idx, images in enumerate(pbar):
            optimizer.zero_grad()
            images = images.float().to(device)

            # Sample random noise
            noise = torch.randn_like(images).to(device)

            # Sample timestep
            t = torch.randint(0, config.diffusion.num_timesteps, (images.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(images, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            loss.backward()

            # Clip gradients
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            interval_loss += loss.item()

            # Log at specified intervals
            if (batch_idx + 1) % log_interval == 0:
                avg_interval_loss = interval_loss / log_interval
                pbar.set_description(
                    f'Epoch: {epoch_idx} | Batch: {batch_idx + 1}/{num_batches} | Loss: {avg_interval_loss:.4f}'
                )
                wandb.log({"train_loss": avg_interval_loss, "epoch": epoch_idx})
                interval_loss = 0.0  # Reset interval loss

        # Log average loss for the epoch
        avg_epoch_loss = total_loss / num_batches
        print(f'Finished epoch: {epoch_idx} | Average Loss: {avg_epoch_loss:.4f}')
        wandb.log({"average_epoch_loss": avg_epoch_loss, "epoch": epoch_idx})

        # # Sample images
        # if epoch_idx % config.training.sample_interval == 0:
        #     save_path = os.path.join(config.folders.samples, run_id)
        #     sample(model, scheduler, config.training, config.model, config.diffusion, save_path)

        # Save model checkpoints
        if epoch_idx % config.training.save_interval == 0:
            save_checkpoint(epoch_idx, model, optimizer)

    print('Done Training ...')
    wandb.finish()


if __name__ == '__main__':

    # Create Datalaoders
    train_transform = transforms.Compose(
        [
            transforms.Resize(config.data.image_size, transforms.InterpolationMode.BILINEAR
                             ),  # Resize the smallest side to 128 and maintain aspect ratio
            transforms.RandomCrop(config.data.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambd=lambda x: x * 2.0 - 1.0),  # Normalize to [-1, 1]
        ]
    )

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end
    )

    root_dir = Path(config.data.root_dir)

    acdc_image_dir = root_dir / config.data.acdc_images
    dataset = ACDCDataset(acdc_image_dir, config.data.weather, transform=train_transform)

    # add bdd images
    bdd_image_dir = root_dir / config.data.bdd_dir
    dataset.add_images(bdd_image_dir)

    # add dawn images
    dawn_image_dir = root_dir / config.data.dawn_dir
    dataset.add_images(dawn_image_dir)
    print(dataset.__len__())

    data_loader = DataLoader(
        dataset=dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers
    )

    # Instantiate the model
    model = Unet(config.model).to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=config.training.lr)
    criterion = torch.nn.MSELoss()

    setup_logger()
    try:
        train(data_loader, model, optimizer, criterion, scheduler)
    except Exception as e:
        print(e)

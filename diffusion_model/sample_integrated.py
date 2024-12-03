import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from datetime import datetime
from diffusion_model.models.old_modules import UNet
from diffusion_model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from diffusion_model.config.models import TrainingConfig, ModelConfig, DiffusionConfig, FolderConfig, Config
from diffusion_model.train_ddpm import load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)


def save_images(xt, save_path, num_grid_rows):
    grid = make_grid(xt, nrow=num_grid_rows)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(
        os.path.join(save_path, f'old_x_{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second}.png')
    )
    img.close()


def postprocess(xt, mean=[0.4865, 0.4998, 0.4323], std=[0.2326, 0.2276, 0.2659]):
    mean = torch.tensor(mean, device=xt.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=xt.device).view(1, -1, 1, 1)
    images = xt * std + mean
    images = (images * 255).clamp(0, 255).type(torch.uint8).detach().cpu()
    return images


def sample(
    model: UNet,
    scheduler: LinearNoiseScheduler,
    train_config: TrainingConfig,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    save_path: str = 'diffusion_model_v2/outputs/samples'
):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((train_config.sample_size, model_config.im_channels, model_config.im_size, model_config.im_size)
                    ).to(device)

    for i in tqdm(reversed(range(diffusion_config.num_timesteps))):
        # Get prediction of noise
        t = torch.full((xt.size(0),), i, dtype=torch.long).to(device)

        # Predict noise
        noise_pred = model(xt, scheduler.one_minus_cum_prod[t].view(-1, 1, 1, 1))

        mean, sigma, _ = scheduler.sample_prev_timestep2(xt, noise_pred, t)

        xt = mean + sigma if i != 0 else mean

    images = postprocess(xt)
    save_images(images, save_path, train_config.num_grid_rows)


def load_model(model_path: str) -> torch.nn.Module:
    model = UNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_scheduler(diffusion_config: DiffusionConfig) -> LinearNoiseScheduler:
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config.num_timesteps,
        beta_start=diffusion_config.beta_start,
        beta_end=diffusion_config.beta_end
    )
    return scheduler


def infer(config: Config):
    # Load model with checkpoint
    checkpoint_path = os.path.join(config.folders.checkpoints, f'old_model/1000-checkpoint.ckpt')
    model = load_model(checkpoint_path)

    # Load scheduler
    scheduler = load_scheduler(config.diffusion)
    save_path = os.path.join(config.folders.samples, 'old_model')

    with torch.no_grad():
        sample(model, scheduler, config.training, config.model, config.diffusion, save_path)


if __name__ == '__main__':
    config = load_config('diffusion_model/config/config.yaml')

    infer(config)

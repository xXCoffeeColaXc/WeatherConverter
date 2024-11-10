import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm

from diffusion_model.models.unet_base import Unet
from diffusion_model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from diffusion_model.config.models import TrainingConfig, ModelConfig, DiffusionConfig, FolderConfig, Config
from diffusion_model.train_ddpm import load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)


def sample(
    model,
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
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Use scheduler to get x0 and xt-1
        mean, sigma, x0 = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        xt = mean + sigma

    # Save x0
    ims = torch.clamp(x0, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, nrow=train_config.num_grid_rows)
    img = torchvision.transforms.ToPILImage()(grid)
    #img = torchvision.transforms.ToPILImage()(ims[0])
    img.save(os.path.join(save_path, 'x9.png'))
    img.close()


def load_model(model_path: str, model_config: ModelConfig) -> torch.nn.Module:
    model = Unet(model_config).to(device)
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
    epoch = 20
    # Load model with checkpoint
    checkpoint_path = os.path.join(config.folders.checkpoints, f'{epoch}-checkpoint.ckpt')
    model = load_model(checkpoint_path, config.model)

    # Load scheduler
    scheduler = load_scheduler(config.diffusion)
    save_path = config.folders.samples

    with torch.no_grad():
        try:
            sample(model, scheduler, config.training, config.model, config.diffusion, save_path)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    epoch = 20
    config = load_config('diffusion_model_v2/config/config.yaml')
    checkpoint_path = os.path.join(config.folders.checkpoints, f'{epoch}-checkpoint.ckpt')
    model = load_model(checkpoint_path, config.model)
    scheduler = load_scheduler(config.diffusion)
    sample(model, scheduler, config.training, config.model, config.diffusion, config.folders.samples)

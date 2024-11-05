import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from config.models import TrainingConfig, ModelConfig, DiffusionConfig, FolderConfig, Config

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
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

    # Save x0
    ims = torch.clamp(xt, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, nrow=train_config.num_grid_rows)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(os.path.join(save_path, 'x0_{}.png'.format(i)))
    img.close()


def infer(config: Config):
    run_id = 3
    epoch = 60

    # Load model with checkpoint
    model = Unet(config.model).to(device)
    model.load_state_dict(
        torch.load(os.path.join(config.folders.checkpoints, run_id, f'{epoch}-checkpoint.ckpt'), map_location=device)
    )
    model.eval()

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end
    )
    with torch.no_grad():
        sample(model, scheduler, config.training, config.model, config.diffusion, config.folders.samples, run_id)


if __name__ == '__main__':
    config = load_config('diffusion_model_v2/config/config.yaml')
    infer(config)

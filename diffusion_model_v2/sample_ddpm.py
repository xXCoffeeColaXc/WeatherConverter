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
        xt, x0 = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

    # Save x0
    ims = torch.clamp(x0, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, nrow=train_config.num_grid_rows)
    img = torchvision.transforms.ToPILImage()(grid)
    #img = torchvision.transforms.ToPILImage()(ims[0])
    img.save(os.path.join(save_path, 'x9.png'))
    img.close()


def infer(config: Config):
    run_id = 3
    epoch = 380

    # Load model with checkpoint
    model = Unet(config.model).to(device)
    checkpoint_path = os.path.join(config.folders.checkpoints, f'{epoch}-checkpoint.ckpt')
    print(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end
    )
    save_path = config.folders.samples

    with torch.no_grad():
        try:
            sample(model, scheduler, config.training, config.model, config.diffusion, save_path)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    config = load_config('diffusion_model_v2/config/config.yaml')
    infer(config)

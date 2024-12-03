import torch
import torchvision
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from torchvision import transforms
from matplotlib import pyplot as plt
from datetime import datetime
from PIL import Image
from diffusion_model.models.old_modules import UNet
from diffusion_model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from diffusion_model.config.models import TrainingConfig, ModelConfig, DiffusionConfig, FolderConfig, Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)


def save_images(xt, save_path, filename):
    os.makedirs(save_path, exist_ok=True)
    img = torchvision.transforms.ToPILImage()(xt)
    img.save(os.path.join(save_path, filename))
    img.close()


def postprocess(xt, mean=[0.4865, 0.4998, 0.4323], std=[0.2326, 0.2276, 0.2659]):
    mean = torch.tensor(mean, device=xt.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=xt.device).view(1, -1, 1, 1)
    images = xt * std + mean
    images = (images * 255).clamp(0, 255).type(torch.uint8).detach().cpu()
    return images


def visualize_forward_process(scheduler: LinearNoiseScheduler, x0, save_path, num_steps=1000, save_every=100):
    """
    Visualizes the forward diffusion process by adding noise to x0 over num_steps.
    Saves images every save_every steps and concatenates them vertically.
    """
    # Ensure x0 has shape (1, C, H, W)
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)
    elif x0.dim() == 4:
        pass
    else:
        raise ValueError("x0 must have 3 or 4 dimensions")

    x0 = x0.to(device)
    images = []

    for t in range(num_steps):
        t_tensor = torch.full((x0.size(0),), t, dtype=torch.long).to(device)
        #t_tensor = torch.tensor([t]).to(device)

        noise = torch.randn_like(x0).to(device)
        xt = scheduler.add_noise2(x0, noise, t_tensor)

        if (t + 1) % save_every == 0:
            # Postprocess xt
            xt_post = postprocess(xt)
            images.append(xt_post.squeeze(0))  # Remove batch dimension

    # Concatenate images vertically
    concatenated_image = torch.cat(images, dim=2)  # Concatenate along height (dim=1)

    # Save concatenated image
    save_images(concatenated_image, save_path, 'forward_process.png')


def visualize_backward_process(
    model: UNet,
    scheduler: LinearNoiseScheduler,
    model_config: ModelConfig,
    save_path: str,
    num_steps=1000,
    save_every=100
):
    """
    Visualizes the backward diffusion process by denoising from random noise over num_steps.
    Saves images every save_every steps and concatenates them vertically.
    """
    xt = torch.randn((1, model_config.im_channels, model_config.im_size, model_config.im_size)).to(device)
    images = []

    for i in tqdm(reversed(range(num_steps))):
        # Get prediction of noise
        t = torch.full((xt.size(0),), i, dtype=torch.long).to(device)

        # Predict noise
        noise_pred = model(xt, scheduler.one_minus_cum_prod[t].view(-1, 1, 1, 1))

        mean, sigma, _ = scheduler.sample_prev_timestep2(xt, noise_pred, t)

        xt = mean + sigma if i != 0 else mean

        if (num_steps - i) % save_every == 0:
            # Postprocess xt
            xt_post = postprocess(xt)
            images.append(xt_post.squeeze(0))  # Remove batch dimension

    # Concatenate images vertically
    concatenated_image = torch.cat(images, dim=2)  # Concatenate along height (dim=2)

    # Save concatenated image
    save_images(concatenated_image, save_path, 'backward_process.png')


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
    checkpoint_path = os.path.join(config.folders.checkpoints, 'old_model/1000-checkpoint.ckpt')
    model = load_model(checkpoint_path)

    # Load scheduler
    scheduler = load_scheduler(config.diffusion)
    save_path = os.path.join(config.folders.samples, 'old_model')

    transform = transforms.Compose(
        [
            transforms.Resize(128, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4865, 0.4998, 0.4323], std=[0.2326, 0.2276, 0.2659])
        ]
    )
    x0 = transform(
        Image.open(
            '/media/talmacsi/48a93eb4-f27d-48ec-9f74-64e475c3b6ff/Downloads/data/ACDC/rgb_anon/rain/train/GP010400/GP010400_frame_000189_rgb_anon.png'
        ).convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        # Visualize Forward Process
        visualize_forward_process(scheduler, x0, save_path)

        # Visualize Backward Process
        visualize_backward_process(model, scheduler, config.model, save_path)


def apply_photometric_augmentations(image):
    """
    Applies photometric augmentations to an image.
    """
    augmentations = [
        ('Brightness Adjustment', transforms.ColorJitter(brightness=0.5)),  # Adjust brightness
        ('Contrast Adjustment', transforms.ColorJitter(contrast=0.5)),  # Adjust contrast
        ('Saturation Adjustment', transforms.ColorJitter(saturation=0.5)),  # Adjust saturation
        ('Hue Adjustment', transforms.ColorJitter(hue=0.3)),  # Adjust hue
    ]
    augmented_images = []
    for name, transform in augmentations:
        aug_image = transform(image)
        augmented_images.append((name, aug_image))
    return augmented_images


def apply_geometric_augmentations(image):
    """
    Applies geometric augmentations to an image.
    """
    augmentations = [
        ('Affine Rotation 30°', transforms.RandomAffine(degrees=30)),  # Rotate up to 30 degrees
        ('Affine Translation', transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))),  # Translate up to 10%
        ('Affine Scaling', transforms.RandomAffine(degrees=0, scale=(0.1, 1.5))),  # Scale between 80% to 120%
        ('Affine Shear', transforms.RandomAffine(degrees=0, shear=50)),  # Shear up to 20 degrees
    ]
    augmented_images = []
    for name, transform in augmentations:
        aug_image = transform(image)
        augmented_images.append((name, aug_image))
    return augmented_images


# Define function to visualize augmentations
def visualize_augmentations(original_image, augmented_images_list, category_name):
    """
    Visualizes original and augmented images.
    """
    num_augmentations = len(augmented_images_list)
    fig, axes = plt.subplots(1, num_augmentations + 1, figsize=((num_augmentations + 1) * 3, 3))
    fig.suptitle(f'{category_name} Augmentations', fontsize=16)

    # Original image in the first subplot
    img = original_image.permute(1, 2, 0).numpy()
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Augmented images
    for i, (name, aug_image) in enumerate(augmented_images_list):
        img = aug_image.permute(1, 2, 0).numpy()
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(name)
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{category_name}_augmentations.png')


if __name__ == '__main__':
    #config = load_config('diffusion_model/config/config.yaml')
    #infer(config)

    num_images = 4
    image_size = 128
    image = transforms.PILToTensor()(
        Image.open(
            '/media/talmacsi/48a93eb4-f27d-48ec-9f74-64e475c3b6ff/Downloads/data/ACDC/rgb_anon/fog/train/GP020478/GP020478_frame_000320_rgb_anon.png'
        ).convert("RGB")
    )

    # Apply photometric augmentations
    photometric_augmented_images = apply_photometric_augmentations(image)
    visualize_augmentations(image, photometric_augmented_images, 'Photometric')

    # Apply geometric augmentations
    geometric_augmented_images = apply_geometric_augmentations(image)
    visualize_augmentations(image, geometric_augmented_images, 'Geometric')

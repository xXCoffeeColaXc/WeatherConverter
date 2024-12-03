import torch
import numpy as np
import os
from pathlib import Path
import diffusion_model.sample_integrated as ddpm
from tqdm import tqdm
from diffusion_model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
import seg_model.inference as seg_infer
import srgan_model.inference as srgan_infer
from sgg.sgg import apply_gsg, apply_lcg
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def debug_tensor(tensor: torch.Tensor, path: str, title: str = None):
    if title:
        print(title)

    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor min: {tensor.min().item()}")
    print(f"Tensor max: {tensor.max().item()}")
    print(f"Tensor device: {tensor.device}")

    if tensor.ndim == 4 and tensor.shape[1] == 3:
        tensor = tensor.squeeze(0)  # [3, 128, 128]
        tensor = torch.clamp(tensor, -1., 1.).detach().cpu()
        tensor = (tensor + 1) / 2
    elif tensor.ndim == 3 and tensor.shape[0] == 1:  # [1, 512, 512]
        tensor = tensor.float()
        print(f"Tensor unique values: {tensor.unique()}")

    # Convert the tensor to a PIL Image
    pil_image = transforms.ToPILImage()(tensor)
    pil_image.save(path)

    print(f"Image saved to {path}")
    print('-' * 50)


def visualize_result(output_tensor: torch.Tensor, input_tensor: torch.Tensor, lbl_colored_img: torch.Tensor):
    pass


def sample_with_sgg(
    input_tensor: torch.Tensor,  # [1, 3, 128, 128] normalized to [-1, 1]
    diff_model: torch.nn.Module,
    diff_scheduler: LinearNoiseScheduler,
    seg_model: torch.nn.Module,
    gt: torch.Tensor,  # [1, 512, 512] encoded label (0-18 values)
    gt_128: torch.Tensor,  # [1, 128, 128] encoded label (0-18 values)
    srgan_model: torch.nn.Module,
) -> torch.Tensor:

    LAMBDA = 60.0
    N = 100

    #debug_tensor(input_tensor, 'debug/input.png', 'input_tensor')
    #debug_tensor(gt, 'debug/gt.png', 'gt')

    # --------- FORWARD PROCESS ------------
    # add noise to input image for N steps
    #t = torch.randint(0, N, (input_tensor.shape[0],)).to(device)
    t_n = torch.full((input_tensor.size(0),), N, dtype=torch.long).to(device)
    print(f"t: {t_n}")

    noise = torch.randn_like(input_tensor).to(device)
    xt = diff_scheduler.add_noise2(input_tensor, noise, t_n)

    debug_tensor(xt, f'debug/xt_{N}_noised.png', 'xt_noised')

    # --------- REVERSE PROCESS ------------
    for i in tqdm(reversed(range(N))):
        #print(f"Step {i}")
        t = torch.full((xt.size(0),), i, dtype=torch.long).to(device)
        xt = xt.float()

        # Get prediction of noise
        noise_pred = diff_model(xt, diff_scheduler.one_minus_cum_prod[t].view(-1, 1, 1, 1))
        #debug_tensor(noise_pred, f'debug/noise_pred_{i}.png', 'noise_pred')

        # Use scheduler to get mu and sigma
        mu, sigma, _ = diff_scheduler.sample_prev_timestep2(xt, noise_pred, t)

        # Upscale xt to 512x512
        sr_xt = srgan_infer.inference(srgan_model, xt)

        # Apply SGG
        if i % 2 == 0 and i != 0:
            xt = apply_lcg(seg_model, mu, sigma, sr_xt, gt, gt_128, LAMBDA)
        elif i % 2 == 1 and i != 0:
            xt = apply_gsg(seg_model, mu, sigma, sr_xt, gt, LAMBDA)

        if i == 0:
            xt = mu

        #debug_tensor(xt, f'debug/xt_{i}.png', 'xt')

    # Upscale x0 to 512x512
    sr_x0 = srgan_infer.inference(srgan_model, xt)

    return sr_x0


if __name__ == '__main__':

    # ----------------------------------------------------------------------------
    # ------------------------------ Load Diffusion Model ------------------------
    # ----------------------------------------------------------------------------
    diff_config = ddpm.load_config('diffusion_model/config/config.yaml')
    diff_checkpoint_path = os.path.join(diff_config.folders.checkpoints, f'old_model/1000-checkpoint.ckpt')
    diff_model = ddpm.load_model(diff_checkpoint_path)
    diff_scheduler = ddpm.load_scheduler(diff_config.diffusion)

    # ----------------------------------------------------------------------------
    # ------------------------------ Load SRGAN Model ----------------------------
    # ----------------------------------------------------------------------------
    srgan_model_path = '/media/talmacsi/48a93eb4-f27d-48ec-9f74-64e475c3b6ff/Downloads/swift_srgan_4x.pth.tar'
    srgan_model = srgan_infer.load_model(srgan_model_path)

    # ----------------------------------------------------------------------------
    # ------------------------------ Load Segment Model --------------------------
    # ----------------------------------------------------------------------------
    seg_config = seg_infer.load_config('seg_model/config/config.yaml')
    seg_model_path = os.path.join(seg_config.folders.checkpoints, 'augmented/deeplabv3plus_resnet101_epoch_40.pth')
    seg_model = seg_infer.load_model(seg_model_path, seg_config.model)

    # ----------------------------------------------------------------------------
    # ------------------------------ Load Test Data ------------------------------
    # ----------------------------------------------------------------------------
    rgb_anon_path = Path(seg_config.data.root_dir) / seg_config.data.images
    gt_path = Path(seg_config.data.root_dir) / seg_config.data.labels
    gt_label_ids_path = str(gt_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_gt_labelIds.png')  # (0-32 values)
    gt_color_path = str(gt_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_gt_labelColor.png')  # (RGB values)
    input_image_path = str(rgb_anon_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_rgb_anon.png')
    input_image = Image.open(input_image_path).convert("RGB")

    # ----------------------------------------------------------------------------
    # ------------------------------ Preprocess Test Data ------------------------
    # ----------------------------------------------------------------------------
    _, input_tensor_512, encoded_label_tensor_512, encoded_label_tensor_128, lbl_colored_img_512 = seg_infer.preprocess(ori_img_path=input_image_path, img_path=input_image_path, gt_label_ids_path=gt_label_ids_path, gt_color_path=gt_color_path)

    input_transform = transforms.Compose(
        [
            transforms.Resize(diff_config.data.image_size, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(diff_config.data.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambd=lambda x: x * 2.0 - 1.0),  # Normalize to [-1, 1]
        ]
    )
    input_tensor_128: torch.Tensor = input_transform(input_image).unsqueeze(0).to(device)  # [1, 3, 128, 128]

    # ----------------------------------------------------------------------------
    # ------------------------------ Run translation -----------------------------
    # ----------------------------------------------------------------------------

    output_tensor_512 = sample_with_sgg(
        input_tensor=input_tensor_128,
        diff_model=diff_model,
        diff_scheduler=diff_scheduler,
        seg_model=seg_model,
        gt=encoded_label_tensor_512,
        gt_128=encoded_label_tensor_128,
        srgan_model=srgan_model
    )
    debug_tensor(output_tensor_512, 'debug/output_512.png', 'output_512')

    #visualize_result(output_tensor_512, input_tensor_512, lbl_colored_img_512)

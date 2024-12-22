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
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN, STD = [0.4865, 0.4998, 0.4323], [0.2326, 0.2276, 0.2659]


# For the diffusion model
def normalize_(tensor):
    mean = torch.tensor(MEAN, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(1, -1, 1, 1)
    return (tensor - mean) / std


def denormalize_(tensor):
    mean = torch.tensor(MEAN, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(1, -1, 1, 1)
    return tensor * std + mean


def postprocess(xt, mean=[0.4865, 0.4998, 0.4323], std=[0.2326, 0.2276, 0.2659]):
    mean = torch.tensor(mean, device=xt.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=xt.device).view(1, -1, 1, 1)
    images = xt * std + mean
    images = (images * 255).clamp(0, 255).type(torch.uint8).detach().cpu()
    return images


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
    mode: str = 'gsg',  # 'gsg' or 'lcg' or 'sgg'
    LAMBDA: float = 60.0,
    N: int = 400
) -> torch.Tensor:

    print('Starting translation with mode:', mode)

    # --------- FORWARD PROCESS ------------
    # add noise to input image for N steps
    t_n = torch.full((input_tensor.size(0),), N, dtype=torch.long).to(device)

    noise = torch.randn_like(input_tensor).to(device)
    xt = diff_scheduler.add_noise2(input_tensor, noise, t_n)

    # --------- REVERSE PROCESS ------------
    for i in tqdm(reversed(range(N))):
        t = torch.full((xt.size(0),), i, dtype=torch.long).to(device)
        xt = xt.float()

        # Get prediction of noise
        with torch.no_grad():
            noise_pred = diff_model(xt, diff_scheduler.one_minus_cum_prod[t].view(-1, 1, 1, 1))

        # Use scheduler to get mu and sigma
        mu, sigma, _ = diff_scheduler.sample_prev_timestep2(xt, noise_pred, t)

        if i != 0:
            xt = mu + sigma
            xt_denorm = denormalize_(xt).clamp(0, 1)  # denormalize to [0, 1]
            sr_xt = srgan_infer.inference(srgan_model, xt_denorm)  # Upscale xt to 512x512
            sr_xt_norm = normalize_(sr_xt)

            if mode == 'gsg':
                xt = apply_gsg(seg_model, mu, sigma, sr_xt_norm, gt, LAMBDA, True, i)
            elif mode == 'lcg':
                xt = apply_lcg(seg_model, mu, sigma, sr_xt_norm, gt, gt_128, LAMBDA, False)
            elif mode == 'sgg':
                if i % 2 == 0:
                    xt = apply_lcg(seg_model, mu, sigma, sr_xt_norm, gt, gt_128, LAMBDA)
                elif i % 2 == 1:
                    xt = apply_gsg(seg_model, mu, sigma, sr_xt_norm, gt, LAMBDA)
        else:
            xt = mu

        xt = xt.detach()

    # Upscale x0 to 512x512
    xt_denorm = denormalize_(xt).clamp(0, 1)
    final_output = srgan_infer.inference(srgan_model, xt_denorm)

    save_image(final_output, f'debug/final_output_{N}_{LAMBDA}_{mode}.png')

    return xt


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
    seg_model_path = os.path.join(seg_config.folders.checkpoints, 'augmented/deeplabv3plus_resnet101_epoch_80.pth')
    seg_model = seg_infer.load_model(seg_model_path, seg_config.model)

    # ----------------------------------------------------------------------------
    # ------------------------------ Load Test Data ------------------------------
    # ----------------------------------------------------------------------------
    rgb_anon_path = Path(seg_config.data.root_dir) / seg_config.data.images
    gt_path = Path(seg_config.data.root_dir) / seg_config.data.labels
    gt_label_ids_path = str(gt_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_gt_labelIds.png')  # (0-32 values)
    gt_color_path = str(gt_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_gt_labelColor.png')  # (RGB values)
    input_image_path = 'srgan_model/image_super_resolved.png'
    input_image = Image.open(input_image_path).convert("RGB")

    # ----------------------------------------------------------------------------
    # ------------------------------ Preprocess Test Data ------------------------
    # ----------------------------------------------------------------------------
    _, input_tensor_512, encoded_label_tensor_512, encoded_label_tensor_128, lbl_colored_img_512 = seg_infer.preprocess(ori_img_path=input_image_path, img_path=input_image_path, gt_label_ids_path=gt_label_ids_path, gt_color_path=gt_color_path)

    input_transform = transforms.Compose(
        [
            transforms.Resize((1080 // 8, 1920 // 8)),
            transforms.CenterCrop(diff_config.data.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
            #transforms.Lambda(lambd=lambda x: x * 2.0 - 1.0),  # Normalize to [-1, 1]
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
        srgan_model=srgan_model,
        mode='gsg',  # gsg, lcg, sgg, diff
        LAMBDA=100.0,
        N=300
    )
    #debug_tensor(output_tensor_512, 'debug/output_512.png', 'output_512')

    #visualize_result(output_tensor_512, input_tensor_512, lbl_colored_img_512)

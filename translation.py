import torch
import os
from pathlib import Path
import diffusion_model.sample_ddpm as ddpm
from tqdm import tqdm
from diffusion_model.scheduler.linear_noise_scheduler import LinearNoiseScheduler
import seg_model.inference as seg_infer
import srgan_model.inference as srgan_infer
from sgg.sgg import apply_gsg, apply_lcg
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def debug_tensor(tensor: torch.Tensor, path: str):
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor min: {tensor.min()}")
    print(f"Tensor max: {tensor.max()}")
    print(f"Tensor device: {tensor.device}")

    # Save tensor as image
    tensor = tensor.squeeze(0).cpu()
    tensor = torch.clamp(tensor, 0, 1)
    tensor: Image.Image = transforms.ToPILImage()(tensor)
    tensor.save(path)


def visualize_result(x0: torch.Tensor, input_tensor: torch.Tensor, lbl_colored_img: torch.Tensor):
    pass


def sample_with_sgg(
    input_tensor: torch.Tensor,
    diff_model: torch.nn.Module,
    scheduler: LinearNoiseScheduler,
    diffusion_config: ddpm.Config,
    seg_model: torch.nn.Module,
    gt: torch.Tensor,
    srgan_model: torch.nn.Module,
) -> torch.Tensor:

    lambda_gsg = 60.0
    lambda_lcg = 60.0
    N = 500  # 1000 steps (400 foward, 600 backward)

    x_noise = torch.randn(
        (
            diffusion_config.training.sample_size,
            diffusion_config.model.im_channels,
            diffusion_config.model.im_size,
            diffusion_config.model.im_size
        )
    ).to(device)
    debug_tensor(x_noise, 'debug/x_noise.png')

    # --------- FORWARD PROCESS ------------
    # add noise to input image for N steps
    t = torch.randint(0, N, (input_tensor.shape[0],)).to(device)
    noise = torch.randn_like(input_tensor).to(device)
    xt = scheduler.add_noise(xt, noise, t)

    debug_tensor(xt, f'debug/xt_{N}_noised.png')

    # --------- REVERSE PROCESS ------------
    for i in tqdm(reversed(range(N))):
        print(f"Step {i}")

        # Get prediction of noise
        noise_pred = diff_model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        debug_tensor(noise_pred, f'debug/noise_pred_{i}.png')

        # Use scheduler to get x0 and xt-1
        mean, sigma, x0 = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # # Upscale x to 512x512
        # sr_xt = srgan_infer.inference(srgan_model, xt)

        # # Apply SGG (returns 128x128 scaled gradients)
        # if i % 2 == 0:
        #     grads = apply_lcg(seg_model, sr_xt, gt, lambda_lcg)
        # elif i % 2 == 1:
        #     grads = apply_gsg(seg_model, sr_xt, gt, lambda_gsg)

        # Update xt with the gradients
        #xt = mean * grads + sigma
        xt = mean + sigma
        debug_tensor(xt, f'debug/xt_{i}.png')

    # Upscale x0 to 512x512
    sr_x0 = srgan_infer.inference(srgan_model, xt)

    return sr_x0


if __name__ == '__main__':

    # ---------- Load Diffusion Model ----------
    diff_config = ddpm.load_config('diffusion_model_v2/config/config.yaml')
    diff_checkpoint_path = os.path.join(diff_config.folders.checkpoints, f'20-checkpoint.ckpt')
    diff_model = ddpm.load_model(diff_checkpoint_path, diff_config.model)
    diff_scheduler = ddpm.load_scheduler(diff_config.diffusion)

    # ---------- Load SRGAN Model ----------
    srgan_model_path = 'srgan_model/outputs/checkpoints/srgan_epoch_100.pth'
    srgan_model = srgan_infer.load_model(srgan_model_path)

    # ---------- Load Segment Model ----------
    seg_config = seg_infer.load_config('seg_model/config/config.yaml')
    seg_model_path = os.path.join(seg_config.folders.checkpoints, 'deeplabv3plus_resnet101_epoch_28.pth')
    seg_model = seg_infer.load_model(seg_model_path, seg_config.model)

    # ---------- Load Test Data ----------
    rgb_anon_path = Path(seg_config.data.root_dir) / seg_config.data.images
    gt_path = Path(seg_config.data.root_dir) / seg_config.data.labels
    gt_label_ids_path = str(gt_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_gt_labelIds.png')
    gt_color_path = str(gt_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_gt_labelColor.png')
    input_image_path = str(rgb_anon_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_rgb_anon.png')
    input_image = Image.open(input_image_path).convert("RGB")

    # ---------- Preprocess Test Data ----------
    input_tensor_512, decoded_label_tensor_512, lbl_colored_img_512 = seg_infer.preprocess(input_image_path, gt_label_ids_path, gt_color_path)
    val_transform = transforms.Compose(
        [
            transforms.Resize(diff_config.data.image_size, transforms.InterpolationMode.BILINEAR
                             ),  # Resize the smallest side to 128 and maintain aspect ratio
            transforms.RandomCrop(diff_config.data.image_size),
            transforms.ToTensor(),  #transforms.Lambda(lambd=lambda x: x * 2.0 - 1.0),  # Normalize to [-1, 1]
        ]
    )
    input_tensor_128: torch.Tensor = val_transform(input_image).unsqueeze(0).to(device)  # [1, 3, 128, 128]

    # ---------- Run translation ----------
    try:
        output_tensor_512 = sample_with_sgg(
            input_tensor=input_tensor_128,
            diff_model=diff_model,
            diff_scheduler=diff_scheduler,
            diff_config=diff_config,
            seg_model=seg_model,
            gt=decoded_label_tensor_512,
            srgan_model=srgan_model
        )
    except Exception as e:
        print(e)

    debug_tensor(output_tensor_512, 'debug/output_512.png')

    #visualize_result(output_tensor_512, input_tensor_512, lbl_colored_img_512)

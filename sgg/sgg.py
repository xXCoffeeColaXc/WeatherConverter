import torch
import torch.nn as nn
import torch.nn.functional as F
from seg_model.inference import infer, compute_gradient_magnitude
from torchvision.utils import save_image
from seg_model.datasets.acdc import ACDCDataset
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN, STD = [0.4865, 0.4998, 0.4323], [0.2326, 0.2276, 0.2659]


def apply_gsg(
    seg_model: nn.Module,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    sr_xt_norm_seg: torch.Tensor,
    gt: torch.Tensor,
    _lambda: float,
    debug_images: bool = False,
    iteration: int = 0
) -> torch.Tensor:
    # Adjust mu based on global guidance
    # L(global)[xt, y] = L(ce)[g(xt), y]
    # mu_hat(xt, t) = mu(xt, t) + lambda * sigma * gradient(L(global)[xt,y])

    sr_xt_norm_seg.requires_grad_(True)

    # Compute gradients w.r.t sr_xt_norm_seg
    pred, input_gradients_seg, _ = infer(seg_model, sr_xt_norm_seg, gt)  # gradients are in seg model normalization space

    # Downscale gradients to match mu's spatial dimensions
    gradients_avg_128 = F.interpolate(input_gradients_seg, size=mu.shape[-2:], mode='bilinear', align_corners=False)

    if debug_images:
        colorized_preds = ACDCDataset.decode_target(pred).astype('uint8')
        colorized_preds = Image.fromarray(colorized_preds)
        colorized_preds.save(f'debug/gsg/aug/colorized_preds_at_{str(iteration)}_lambda_{_lambda}.png')

    mu_hat = mu + _lambda * sigma * gradients_avg_128
    xt = mu_hat + sigma

    return xt.detach()


def apply_lcg(
    seg_model: nn.Module,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    sr_xt_norm_seg: torch.Tensor,
    gt: torch.Tensor,
    gt_128: torch.Tensor,
    _lambda: float,
    debug_images: bool = False
) -> torch.Tensor:
    # L(local)[xt, y, c] = L(ce)([g(xt * mc)], y*mc)
    # mu_hat(xt, t, c) = mu(xt, t) + lambda * sigma * gradient(L(local)[xt, y, c])
    # xt_c = mu_hat + sigma
    # xt = sum_c(mc * xt_c)

    # Adjust mu based on local guidance
    xt_c_list = []
    mc_list = []

    xt = torch.zeros_like(mu)
    mc_total = torch.zeros_like(mu[:, :1, :, :])  # Initialize total mask

    if debug_images:
        save_image(sr_xt_norm_seg, 'debug/lcg/aug/sr_xt_norm_seg.png')

    for c in range(19):  # 19 classes

        # Generate class-specific mask
        mc = (gt == c).long().unsqueeze(1).to(device)  # [1,1,512,512]
        if (mc.sum() == 0):
            continue  # Skip classes not present in the image
        mc_128 = (gt_128 == c).long().unsqueeze(1).to(device)  # [1,1,128,128]

        mc_list.append(mc_128.detach())

        sr_xt_masked = sr_xt_norm_seg * mc  # [1,3,512,512]
        gt_masked = gt * mc.squeeze(1)  # [1,512,512]

        sr_xt_masked.requires_grad_(True)

        pred, input_gradients_seg, _ = infer(seg_model, sr_xt_masked, gt_masked.long()) # gradients are in seg model normalization space

        # Downscale gradients
        gradients_avg_128 = F.interpolate(input_gradients_seg, size=mu.shape[-2:], mode='bilinear', align_corners=False)

        mu_hat_c = mu + _lambda * sigma * gradients_avg_128
        xt_c = mu_hat_c + sigma

        xt_c_list.append(xt_c.detach())

        if debug_images:
            # Visualize mc (mask for class c)
            mc_np = mc.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8) * 255  # Multiply by 255 to get [0, 255]
            #mc_image = Image.fromarray(mc_np)
            #mc_image.save(f'debug/lcg/aug/mc512_class_{c}.png')

            # Visualize mc_128
            #mc_128_np = mc_128.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8) * 255
            #mc_128_image = Image.fromarray(mc_128_np)
            #mc_128_image.save(f'debug/lcg/aug/mc128_class_{c}.png')

            colorized_preds = ACDCDataset.decode_target(pred).astype('uint8')
            colorized_preds_masked = colorized_preds * mc_np.reshape(512, 512, 1)
            colorized_preds_masked = Image.fromarray(colorized_preds_masked)
            colorized_preds_masked.save(f'debug/lcg/aug/preds_class_{c}.png')

            mean_seg = torch.tensor(MEAN, device=sr_xt_masked.device).view(1, -1, 1, 1)
            std_seg = torch.tensor(STD, device=sr_xt_masked.device).view(1, -1, 1, 1)
            sr_xt_masked_denorm = sr_xt_masked * std_seg + mean_seg  # De-normalize
            sr_xt_masked_denorm = sr_xt_masked_denorm.clamp(0, 1)

            # Convert to CPU and numpy
            sr_xt_masked_np = sr_xt_masked_denorm.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            sr_xt_masked_image = Image.fromarray((sr_xt_masked_np * 255).astype(np.uint8))
            sr_xt_masked_image.save(f'debug/lcg/aug/sr_xt_masked_class_{c}.png')

            # Visualize gt_masked
            # gt_masked_np = gt_masked.squeeze(0).cpu().numpy()
            # gt_masked_color = ACDCDataset.decode_target(gt_masked_np).astype('uint8')
            # gt_masked_image = Image.fromarray(gt_masked_color)
            #gt_masked_image.save(f'debug/lcg/aug/gt_masked_class_{c}.png')

        # Add to xt
        xt += xt_c * mc_128
        mc_total += mc_128

    # Add remaining classes
    mc_remaining = 1 - mc_total
    if mc_remaining.sum() > 0:
        xt += (mu + sigma) * mc_remaining

    return xt.detach()

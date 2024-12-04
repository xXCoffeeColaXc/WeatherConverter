import torch
import torch.nn as nn
import torch.nn.functional as F
from seg_model.inference import infer, compute_gradient_magnitude

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def apply_gsg(
    seg_model: nn.Module, mu: torch.Tensor, sigma: torch.Tensor, sr_xt: torch.Tensor, gt: torch.Tensor, _lambda: float
) -> torch.Tensor:
    # Adjust mu based on global guidance
    # L(global)[xt, y] = L(ce)[g(xt), y]
    # mu_hat(xt, t) = mu(xt, t) + lambda * sigma * gradient(L(global)[xt,y])

    _, input_gradients, _ = infer(seg_model, sr_xt, gt)

    gradients_avg_128 = F.avg_pool2d(input_gradients, kernel_size=4, stride=4)
    gradient_magnitude_avg_128 = compute_gradient_magnitude(gradients_avg_128, denormalize=True, norm=False)

    mu_hat = mu + _lambda * sigma * gradient_magnitude_avg_128
    xt = mu_hat + sigma

    # Detach xt to prevent computational graph retention
    xt = xt.detach()

    # Clean up variables to free memory
    del input_gradients, gradients_avg_128, gradient_magnitude_avg_128, mu_hat
    torch.cuda.empty_cache()

    return xt


def apply_lcg(
    seg_model: nn.Module,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    sr_xt: torch.Tensor,
    gt: torch.Tensor,
    gt_128: torch.Tensor,
    _lambda: float
) -> torch.Tensor:
    # L(local)[xt, y, c] = L(ce)([g(xt * mc)], y*mc)
    # mu_hat(xt, t, c) = mu(xt, t) + lambda * sigma * gradient(L(local)[xt, y, c])
    # xt_c = mu_hat + sigma
    # xt = sum_c(mc * xt_c)

    # Adjust mu based on local guidance
    xt_c_list = []
    mc_list = []

    for c in range(19):  # 19 classes
        # Generate class-specific mask
        mc = (gt == c).long().unsqueeze(1).to(device)  # [1,1,512,512]
        mc_128 = F.interpolate(mc.float(), size=(128, 128), mode='nearest').long()
        mc_list.append(mc_128.detach())

        xt_masked = sr_xt * mc  # [1,3,512,512]
        gt_masked = gt * mc.squeeze(0)  # [1,512,512]

        _, input_gradients, _ = infer(seg_model, xt_masked, gt_masked)

        gradients_avg_128 = F.avg_pool2d(input_gradients, kernel_size=4, stride=4)
        gradient_magnitude_avg_128 = compute_gradient_magnitude(gradients_avg_128, denormalize=True, norm=False)

        mu_hat_c = mu + _lambda * sigma * gradient_magnitude_avg_128
        xt_c = mu_hat_c + sigma

        xt_c_list.append(xt_c.detach())

        # Clean up variables to free memory
        del xt_masked, gt_masked, input_gradients, gradients_avg_128, gradient_magnitude_avg_128, mu_hat_c
        torch.cuda.empty_cache()

    # Sum all class-specific xt_c
    xt = torch.sum(torch.stack(xt_c_list) * torch.stack(mc_list), dim=0)

    return xt

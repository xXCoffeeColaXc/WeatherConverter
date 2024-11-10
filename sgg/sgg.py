import torch
import torch.nn as nn
import torch.nn.functional as F
from seg_model.inference import infer, compute_gradient_magnitude

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def apply_gsg(seg_model: nn.Module, x: torch.Tensor, gt: torch.Tensor, lambda_gsg: float) -> torch.Tensor:
    # Adjust mu based on global guidance
    # L(global)[x, y] = L(ce)[g(x), y],
    # mu_hat(x,k) = mu(x,k) + lambda * cov * gradient(L(global)[x,y])

    _, input_gradients, _ = infer(seg_model, x, gt)

    gradients_avg_128 = F.avg_pool2d(input_gradients, kernel_size=4, stride=4)
    gradient_magnitude_avg_128 = compute_gradient_magnitude(gradients_avg_128, denormalize=True, norm=False)

    # Clip gradients ?

    return gradient_magnitude_avg_128 * lambda_gsg


def apply_lcg(seg_model: nn.Module, xt: torch.Tensor, gt: torch.Tensor, lambda_lcg: float) -> torch.Tensor:
    # Adjust mu based on local guidance
    gradients_sum = torch.zeros_like(xt)

    for c in range(19):  # 19 classes
        # Generate class-specific mask
        mc = (gt == c).long().unsqueeze(1).to(device)  # [1,1,512,512]

        xt_masked = xt * mc  # [1,3,512,512]
        gt_masked = gt * mc.squeeze(0)  # [1,512,512]

        _, input_gradients, _ = infer(seg_model, xt_masked, gt_masked)

        gradients_avg_128 = F.avg_pool2d(input_gradients, kernel_size=4, stride=4)
        gradient_magnitude_avg_128 = compute_gradient_magnitude(gradients_avg_128, denormalize=True, norm=False)

        # mask gradients
        gradient_magnitude_avg_128 = gradient_magnitude_avg_128 * mc.squeeze(0).cpu().numpy()

        gradients_sum += gradient_magnitude_avg_128

    return gradients_sum * lambda_lcg

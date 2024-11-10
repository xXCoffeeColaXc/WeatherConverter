import torch
import numpy as np
from PIL import Image
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms as T

import seg_model.network as network
from seg_model.datasets.acdc import ACDCDataset
from seg_model.utils.utils import Denormalize

from seg_model.config.models import Config, ModelConfig
from seg_model.utils.ext_transforms import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)


# Load the model
def load_model(model_path: str, model_config: ModelConfig) -> torch.nn.Module:
    model: torch.nn.Module = network.modeling.__dict__[
        model_config.name](num_classes=model_config.num_classes, output_stride=model_config.output_stride)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def compute_gradient_magnitude(
    input_gradients: torch.Tensor, denormalize: bool = True, norm: bool = False
) -> torch.Tensor:
    gradients_np = input_gradients.squeeze(0).cpu().numpy()  # Shape: [3, 512, 512]

    if denormalize:
        gradients_np = gradients_np * np.array([0.229, 0.224, 0.225])[:, None, None]
    gradient_magnitude = np.sqrt(np.sum(gradients_np**2, axis=0))  # Shape: [512, 512]

    print("Gradient min:", gradient_magnitude.min())
    print("Gradient max:", gradient_magnitude.max())

    if norm:
        # Normalize gradient magnitude to [0,1] for better visualization
        gradient_magnitude = (gradient_magnitude -
                              gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())

    return torch.from_numpy(gradient_magnitude)


def preprocess(img_path: str, gt_label_ids_path: str,
               gt_color_path: str) -> tuple[torch.Tensor, torch.Tensor, Image.Image]:
    # Load your image
    img = Image.open(img_path).convert("RGB")
    label = Image.open(gt_label_ids_path)
    lable_colored = Image.open(gt_color_path)

    val_transform = ExtCompose(
        [
            ExtResize(size=(1080 // 2, 1920 // 2), interpolation=Image.BILINEAR, just_label=True),
            ExtCenterCrop(size=(512, 512), just_label=True),
            ExtToTensor(),
            ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_color_transform = T.Compose(
        [T.Resize(size=(1080 // 2, 1920 // 2), interpolation=Image.BILINEAR), T.CenterCrop(size=(512, 512))]
    )

    # Apply transformations
    input_tensor, lbl_tensor = val_transform(img, label)
    lbl_colored_img = val_color_transform(lable_colored)
    print(type(input_tensor), type(lbl_tensor), type(lbl_colored_img))
    # Prepare tensors for the model
    input_tensor = input_tensor.unsqueeze(0).to(device)

    print(f"Unique values in label tensor: {torch.unique(lbl_tensor)}")

    decoded_label = ACDCDataset.encode_target(lbl_tensor)
    decoded_label_tensor = torch.from_numpy(np.array(decoded_label)).unsqueeze(0).long().to(device)
    print(f"Unique values in label tensor: {torch.unique(decoded_label_tensor)}")

    print(f"Input Tensor Shape: {input_tensor.shape}")
    print(f"Label Tensor Shape: {decoded_label_tensor.shape}")
    print(f"Label Image Shape: {lbl_colored_img.size}")

    return input_tensor, decoded_label_tensor, lbl_colored_img


def infer(model: torch.nn.Module, input_tensor: torch.Tensor,
          decoded_label_tensor: torch.Tensor) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:

    # NOTE: The batch dimension should be 1 !!!
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # Zero gradients
    model.zero_grad()
    if input_tensor.grad is not None:
        input_tensor.grad.zero_()

    # Enable gradient tracking on the input tensor
    input_tensor.requires_grad = True

    print(f"Input Tensor requires gradient: {input_tensor.requires_grad}")
    print(f"Input tensor gradient: {input_tensor.grad}")

    # Perform inference
    output: torch.Tensor = model(input_tensor)  # Output shape: [1, num_classes, 512, 512]
    print(f"Output tensor shape: {output.shape}")
    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()  # Shape: [512, 512]

    print(f"Predicted tensor shape: {pred.shape}")

    loss = criterion(output, decoded_label_tensor.squeeze(1))  # Shape [1, 512, 512]

    loss.backward()

    input_gradients = input_tensor.grad  # Shape: [1, 3, 512, 512]
    gradients_np = input_gradients.detach().cpu().squeeze(0).numpy()

    return pred, input_gradients, gradients_np


def visualize_samples(input_tensor: torch.Tensor, pred: np.ndarray, lbl_colored_img: Image.Image) -> None:
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Convert input tensor to numpy array for visualization
    image_np = denorm(input_tensor.squeeze(0).detach().cpu()).numpy()
    image_np = np.transpose(image_np, (1, 2, 0))  # From [C, H, W] to [H, W, C]
    image_np = np.clip(image_np, 0, 1) * 255
    image_np = image_np.astype(np.uint8)

    colorized_preds = ACDCDataset.decode_target(pred).astype('uint8')
    colorized_preds = Image.fromarray(colorized_preds)

    # Plotting the input image, predictions, and ground truth
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Display input image
    axes[0].imshow(image_np)
    axes[0].set_title('Resized Image')
    axes[0].axis('off')

    # Display colorized predictions
    axes[1].imshow(colorized_preds)
    axes[1].set_title('Colorized Predictions')
    axes[1].axis('off')

    # Display ground truth labels
    axes[2].imshow(lbl_colored_img)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')

    plt.savefig('seg_model/outputs/pred_plot.png')  # TODO: add unique name
    plt.close()
    print("Visualization saved.")


if __name__ == '__main__':

    config = load_config('seg_model/config/config.yaml')

    rgb_anon_path = Path(config.data.root_dir) / config.data.images
    gt_path = Path(config.data.root_dir) / config.data.labels
    img_path = 'srgan_model/image_super_resolved.png'
    gt_label_ids_path = str(gt_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_gt_labelIds.png')
    gt_color_path = str(gt_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_gt_labelColor.png')

    model_path = 'seg_model/outputs/checkpoints/deeplabv3plus_resnet101_epoch_28.pth'
    model = load_model(model_path, config.model)

    input_tensor, decoded_label_tensor, lbl_colored_img = preprocess(img_path, gt_label_ids_path, gt_color_path)

    pred, input_gradients, gradients_np = infer(model, input_tensor, decoded_label_tensor)

    visualize_samples(input_tensor, pred, lbl_colored_img)

    # # gradients is a tensor of shape [1, C, 512, 512]
    # gradients_avg_128 = F.avg_pool2d(input_gradients, kernel_size=4, stride=4)

    # print(f"Gradients 128 shape: {gradients_avg_128.shape}")

    # gradient_magnitude_avg_128 = compute_gradient_magnitude(gradients_avg_128, denormalize=True, norm=False)

    # # Display gradients
    # plt.imshow(gradient_magnitude_avg_128, cmap='viridis')
    # plt.title('Gradients Magnitude')
    # plt.colorbar()
    # plt.axis('off')

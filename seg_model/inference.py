import torch
import numpy as np
from PIL import Image
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms as T
import torch.nn.functional as F

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
def load_model(model_path: Path, model_config: ModelConfig) -> torch.nn.Module:
    model: torch.nn.Module = network.modeling.__dict__[
        model_config.name](num_classes=model_config.num_classes, output_stride=model_config.output_stride)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def compute_gradient_magnitude(
    input_gradients: torch.Tensor, denormalize: bool = True, norm: bool = False, verbose: bool = False
) -> torch.Tensor:
    gradients_np = input_gradients.squeeze(0).cpu().numpy()  # Shape: [3, 512, 512]

    if denormalize:
        gradients_np = gradients_np * np.array([0.229, 0.224, 0.225])[:, None, None]
    gradient_magnitude = np.sqrt(np.sum(gradients_np**2, axis=0))  # Shape: [512, 512]

    if verbose:
        print("Gradient min:", gradient_magnitude.min())
        print("Gradient max:", gradient_magnitude.max())

    if norm:
        # Normalize gradient magnitude to [0,1] for better visualization
        gradient_magnitude = (gradient_magnitude -
                              gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())

    return torch.from_numpy(gradient_magnitude).to(device)


def preprocess(ori_img_path: str,
               img_path: str,
               gt_label_ids_path: str,
               gt_color_path: str,
               verbose=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Image.Image]:
    '''
    Return the original image, input tensor, encoded label tensor (512), encoded label tensor (128) , and the colorized ground truth image
    '''
    # Load your image
    img = Image.open(img_path).convert("RGB")
    ori_img = Image.open(ori_img_path).convert("RGB")
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

    val_label_mask_transform = T.Compose(
        [T.Resize(size=(1080 // 8, 1920 // 8), interpolation=Image.NEAREST), T.CenterCrop(size=(128, 128))]
    )

    val_color_transform = T.Compose(
        [
            T.Resize(size=(1080 // 2, 1920 // 2), interpolation=Image.BILINEAR),
            T.CenterCrop(size=(512, 512))
            #T.UpSample(size=(512, 512), scale_factor=None, mode='bilinear', align_corners=None)
        ]
    )

    val_ori_transform = T.Compose(
        [T.Resize(size=(1080 // 2, 1920 // 2), interpolation=Image.BILINEAR), T.CenterCrop(size=(512, 512))]
    )

    # Apply transformations
    input_tensor, lbl_tensor = val_transform(img, label)  # input image (512) and label (512)
    lbl_tensor_128 = val_label_mask_transform(label)  # label mask (128)
    lbl_colored_img = val_color_transform(lable_colored)  # colorized gt image (512)
    original_image = val_ori_transform(ori_img)  # original image (512)

    # Prepare tensors for the model
    input_tensor = input_tensor.unsqueeze(0).to(device)

    encoded_label = ACDCDataset.encode_target(lbl_tensor)  # Encode the label to [0, num_classes-1]
    encoded_label_tensor = torch.from_numpy(np.array(encoded_label)).unsqueeze(0).long().to(device)

    encoded_label_128 = ACDCDataset.encode_target(lbl_tensor_128)  # Encode the label to [0, num_classes-1]
    encoded_label_tensor_128 = torch.from_numpy(np.array(encoded_label_128)).unsqueeze(0).long().to(device)

    if verbose:
        print("types: ", type(input_tensor), type(lbl_tensor), type(lbl_colored_img))
        print(f"Unique values in label tensor: {torch.unique(lbl_tensor)}")
        print(f"Unique values in label tensor: {torch.unique(encoded_label_tensor)}")
        print(f"Input Tensor Shape: {input_tensor.shape}")
        print(f"Label Tensor Shape: {encoded_label_tensor.shape}")
        print(f"Label Image Shape: {lbl_colored_img.size}")

    return original_image, input_tensor, encoded_label_tensor, encoded_label_tensor_128, lbl_colored_img


def infer(model: torch.nn.Module,
          input_tensor: torch.Tensor,
          encoded_label_tensor: torch.Tensor,
          verbose=False) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    '''
    Returns the predicted segmentation, input gradients, and gradients as numpy array.
    '''

    # NOTE: The batch dimension should be 1 !!!
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # Zero gradients
    model.zero_grad()
    if input_tensor.grad is not None:
        input_tensor.grad.zero_()

    # Enable gradient tracking on the input tensor
    input_tensor.requires_grad = True

    # Perform inference
    output: torch.Tensor = model(input_tensor)  # Output shape: [1, num_classes, 512, 512]

    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()  # Shape: [512, 512]

    loss = criterion(output, encoded_label_tensor.squeeze(1))  # Shape [1, 512, 512]

    loss.backward()

    input_gradients = input_tensor.grad  # Shape: [1, 3, 512, 512]
    gradients_np = input_gradients.detach().cpu().squeeze(0).numpy()

    if verbose:
        print(f"Input Tensor requires gradient: {input_tensor.requires_grad}")
        print(f"Input tensor gradient: {input_tensor.grad}")
        print(f"Output tensor shape: {output.shape}")
        print(f"Predicted tensor shape: {pred.shape}")

    return pred, input_gradients, gradients_np


def visualize_samples(
    original_image: Image.Image,
    input_tensor: torch.Tensor,
    pred: np.ndarray,
    gradient_magnitude_avg_128: np.ndarray,
    encoded_label_tensor: torch.Tensor,
    lbl_colored_img: Image.Image,
    save_path: Path
) -> None:
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Convert input tensor to numpy array for visualization
    image_np = denorm(input_tensor.squeeze(0).detach().cpu()).numpy()
    image_np = np.transpose(image_np, (1, 2, 0))  # From [C, H, W] to [H, W, C]
    image_np = np.clip(image_np, 0, 1) * 255
    image_np = image_np.astype(np.uint8)

    colorized_preds = ACDCDataset.decode_target(pred).astype('uint8')
    colorized_preds = Image.fromarray(colorized_preds)

    encoded_label_tensor = encoded_label_tensor.squeeze(0).cpu().numpy()

    gradient_magnitude_avg_128 = gradient_magnitude_avg_128.cpu().numpy()

    # Define the images to show
    images_to_show = [
        (original_image, 'Original image'),  # asdaaaaaaaaaaaaaaa
        (image_np, 'SRGAN image'),  # asdaaaaaaaaaaaaaaa
        (colorized_preds, 'Colorized Predictions'),  # asdaaaaaaaaaaaaaaa
        (gradient_magnitude_avg_128, 'Gradient Magnitude'),  # asdaaaaaaaaaaaaaaa
        (encoded_label_tensor, 'Encoded Label Tensor'),  # asdaaaaaaaaaaaaaaa
        (lbl_colored_img, 'Ground Truth')  # asdaaaaaaaaaaaaaaa
    ]

    # Plotting the input image, predictions, and ground truth
    fig, axes = plt.subplots(1, len(images_to_show), figsize=(18, 6))

    # Iterate over the images and display them
    for i, (image, title) in enumerate(images_to_show):
        axes[i].imshow(image)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.savefig(save_path / 'sr_pred_gt_noisy_more.png')  # TODO: add unique name
    plt.close()
    print("Visualization saved.")


if __name__ == '__main__':

    config = load_config('seg_model/config/config.yaml')

    rgb_anon_path = Path(config.data.root_dir) / config.data.images
    gt_path = Path(config.data.root_dir) / config.data.labels
    img_path = 'srgan_model/image_super_resolved.png'  # NOTE: This image is a 128x128 image, but upsampled to 512x512, which is not equal to the original image being resized to 512x512
    gt_label_ids_path = str(gt_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_gt_labelIds.png')
    gt_color_path = str(gt_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_gt_labelColor.png')
    original_img_path = str(rgb_anon_path / 'fog/val/GOPR0476/GOPR0476_frame_000854_rgb_anon.png')

    model_path = Path(config.folders.checkpoints) / 'no_augmented/deeplabv3plus_resnet101_epoch_40.pth'
    model = load_model(model_path, config.model)

    original_image, input_tensor, encoded_label_tensor, lbl_colored_img = preprocess(original_img_path, img_path, gt_label_ids_path, gt_color_path)

    pred, input_gradients, gradients_np = infer(model, input_tensor, encoded_label_tensor)

    # gradients is a tensor of shape [1, C, 512, 512]
    gradients_avg_128 = torch.nn.functional.avg_pool2d(input_gradients, kernel_size=4, stride=4)
    gradient_magnitude_avg_128 = compute_gradient_magnitude(gradients_avg_128, denormalize=True, norm=False)

    save_path = Path(config.folders.samples) / 'no_augmented'
    visualize_samples(
        original_image,
        input_tensor,
        pred,
        gradient_magnitude_avg_128,
        encoded_label_tensor,
        lbl_colored_img,
        save_path
    )

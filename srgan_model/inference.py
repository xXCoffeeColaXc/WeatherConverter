import torch
from PIL import Image
import torchvision.transforms as transforms
from models import Generator  # Ensure models.py is in the same directory

# === Configuration ===
model_path = '/media/talmacsi/48a93eb4-f27d-48ec-9f74-64e475c3b6ff/Downloads/swift_srgan_4x.pth.tar'  # Path to your pre-trained Generator model
input_image_path = '/media/talmacsi/48a93eb4-f27d-48ec-9f74-64e475c3b6ff/Downloads/data/rgb_anon/fog/val/GOPR0476/GOPR0476_frame_000854_rgb_anon.png'  # Path to the input low-resolution image
upscale_factor = 4  # Upscale factor used during training (e.g., 4)
crop_size = 128  # Size of the input image (128x128)

# === Device Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load the Pre-trained Generator Model ===
netG = Generator(upscale_factor=upscale_factor).to(device)
checkpoint = torch.load(model_path, map_location=device)
netG.load_state_dict(checkpoint['model'])
netG.eval()  # Set the model to evaluation mode

# === Define Image Transformations ===
transform = transforms.Compose(
    [
        transforms.Resize((1080 // 8, 1920 // 8)),  # Resize the image to 128x128
        transforms.RandomCrop((crop_size, crop_size)),  # Ensure the image is 128x128
        transforms.ToTensor(),  # Convert PIL Image to Tensor [0,1]
    ]
)

# === Load and Preprocess the Input Image ===
input_image = Image.open(input_image_path).convert('RGB')  # Load image and convert to RGB
lr_image = transform(input_image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# === Perform Inference ===
with torch.no_grad():  # Disable gradient computation
    sr_image = netG(lr_image)  # Generate super-resolved image

# === Post-process the Output Image ===

lr_image = lr_image.squeeze(0).cpu()  # Remove batch dimension and move to CPU
lr_image = torch.clamp(lr_image, 0, 1)  # Clamp values to [0,1]
lr_image = transforms.ToPILImage()(lr_image)  # Convert Tensor to PIL Image
print("Original image size:", lr_image.size)

sr_image = sr_image.squeeze(0).cpu()  # Remove batch dimension and move to CPU
sr_image = torch.clamp(sr_image, 0, 1)  # Clamp values to [0,1]
sr_image = transforms.ToPILImage()(sr_image)  # Convert Tensor to PIL Image
print("Super Res image size:", sr_image.size)

# === Save the Super-Resolved Image ===
lr_image.save('srgan_model/image_original.png')
sr_image.save('srgan_model/image_super_resolved.png')

print(f"Super-resolved image saved.")

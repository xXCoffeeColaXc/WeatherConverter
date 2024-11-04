import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import glob


class ACDCDataset(data.Dataset):

    def __init__(self, root_dir, selected_conditions=['rain', 'fog', 'night'], transform=None):
        self.root_dir = root_dir
        self.selected_conditions = selected_conditions
        self.transform = transform

        self.img_paths = []

        self.preprocess()

    def preprocess(self):
        for condition in self.selected_conditions:
            for split in ['train', 'val', 'test']:
                folder_dir = os.path.join(self.root_dir, condition, split)
                # Use glob to find all .jpg and .png files recursively
                pattern = os.path.join(folder_dir, '**', '*.[jp][pn]g')
                img_files = glob.glob(pattern, recursive=True)
                self.img_paths.extend(img_files)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            image_tensor = (image_tensor * 2) - 1
        return image_tensor


def get_loader(image_dir, selected_attrs, image_size=128, batch_size=16, num_workers=4):
    """Build and return a data loader."""

    # Create Datalaoders
    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR
                             ),  # Resize the smallest side to 128 and maintain aspect ratio
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambd=lambda x: x * 2.0 - 1.0),  # Normalize to [-1, 1]
        ]
    )

    dataset = ACDCDataset(root_dir=image_dir, selected_conditions=selected_attrs, transform=train_transform)
    print(dataset.__len__())

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader

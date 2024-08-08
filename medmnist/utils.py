# utils.py

import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

class ToRGB:
    def __call__(self, img):
        if img.shape[0] == 1:  # Only convert if it's a single-channel image
            img = img.repeat(3, 1, 1)
        return img

def get_datasets(data_flag, download, as_rgb, resize):
    info = INFO[data_flag]
    n_channels = 3 if as_rgb else info['n_channels']

    DataClass = getattr(medmnist, info['python_class'])

    transform_list = []
    if resize:
        transform_list.append(transforms.Resize((224, 224)))
    transform_list.append(transforms.ToTensor())
    if as_rgb:
        transform_list.append(ToRGB())
    transform_list.append(transforms.Normalize(mean=[.5], std=[.5]))

    transform = transforms.Compose(transform_list)

    train_dataset = DataClass(split='train', transform=transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=transform, download=download, as_rgb=as_rgb)

    # Debugging: Inspect the first image in the datasets
    print(f"Train dataset first image shape: {train_dataset[0][0].shape}")
    print(f"Val dataset first image shape: {val_dataset[0][0].shape}")
    print(f"Test dataset first image shape: {test_dataset[0][0].shape}")

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

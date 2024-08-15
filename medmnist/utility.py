import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

class ToRGB:
    def __call__(self, img):
        return img.convert("RGB")

class ToGrayscale:
    def __call__(self, img):
        return img.convert("L")

def get_datasets(data_flag, download, as_rgb, resize, model_flag):
    info = INFO[data_flag]
    n_channels = 3 if as_rgb else info['n_channels']

    DataClass = getattr(medmnist, info['python_class'])

    transform_list = []

    if model_flag == 'medclip_vit':
        # MedCLIP-specific preprocessing
        transform_list.append(transforms.Resize((256, 256)))  # MedCLIP expects 256x256 images
        if n_channels == 3:
            transform_list.append(ToGrayscale())  # Convert to grayscale if needed
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))  # Normalize for MedCLIP

    else:
        # Default preprocessing for ViT, ResNet, etc.
        if resize:
            transform_list.append(transforms.Resize((224, 224)))
        if as_rgb and info['n_channels'] == 1:
            transform_list.append(ToRGB())
        transform_list.append(transforms.ToTensor())
        # Normalize using ImageNet means and stds for other models
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    transform = transforms.Compose(transform_list)

    train_dataset = DataClass(split='train', transform=transform, download=download)
    val_dataset = DataClass(split='val', transform=transform, download=download)
    test_dataset = DataClass(split='test', transform=transform, download=download)

    print("Datasets created:")
    print(f"Train dataset: {train_dataset is not None}")
    print(f"Val dataset: {val_dataset is not None}")
    print(f"Test dataset: {test_dataset is not None}")

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
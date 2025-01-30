import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
# from batchnorm import SynchronizedBatchNorm3d, SynchronizedBatchNorm2d  # Import the SyncBN classes
from torch.nn import BatchNorm2d as SynchronizedBatchNorm2d
import torch
from torch.nn import BatchNorm3d as SynchronizedBatchNorm3d
import torch.nn.functional as F # importing thsi for interpolation- need this in vit3d
from torch.utils.data import DataLoader
from medmnist.utils import ToRGB  # Importing directly from medmnist
# from monai.transforms import (
#     Compose,
#     RandFlipd,
#     RandAffined,
#     Rand3DElasticd,
#     RandGaussianNoised,
#     NormalizeIntensityd,
#     ToTensord,
# )

# class ToRGBmine:
#     def __call__(self, img):
#         # return img.convert("RGB")

class ToGrayscale:
    def __call__(self, img):
        return img.convert("L")

# Define the Transform3D class for 3D data (uncomment this when working with vit3d for resnet and other see below)
# class Transform3D:
#     def __init__(self, mul=None):
#         self.mul = mul

#     def __call__(self, voxel):
#         if self.mul == '0.5':
#             voxel = voxel * 0.5
#         elif self.mul == 'random':
#             voxel = voxel * np.random.uniform()

#         # Convert voxel to tensor and ensure it's float type
#         voxel = torch.tensor(voxel, dtype=torch.float32)

#         # Normalize the voxel values
#         # If voxel has multiple channels, normalization should be done per channel
#         if len(voxel.shape) == 3:  # [D, H, W] for grayscale
#             voxel = (voxel - voxel.mean()) / voxel.std()
#             voxel = voxel.unsqueeze(0)  # Add a channel dimension: [1, D, H, W]
#         elif len(voxel.shape) == 4:  # [C, D, H, W] for multi-channel data (like RGB)
#             for c in range(voxel.shape[0]):
#                 voxel[c] = (voxel[c] - voxel[c].mean()) / voxel[c].std()

#         return voxel

#use thsi for resnet18 3d thats because we already multiply voxel in train and eval func so we remove here
class Transform3D:
    def __init__(self, mul=None):
        # Remove or ignore the 'mul' argument since it's not used anymore
        pass

    def __call__(self, voxel):
        # Convert voxel to tensor and ensure it's float type
        voxel = torch.tensor(voxel, dtype=torch.float32)

        # Normalize the voxel values
        if len(voxel.shape) == 3:  # [D, H, W] for grayscale
            voxel = (voxel - voxel.mean()) / voxel.std()
            voxel = voxel.unsqueeze(0)  # Add a channel dimension: [1, D, H, W]
        elif len(voxel.shape) == 4:  # [C, D, H, W] for multi-channel data (like RGB)
            for c in range(voxel.shape[0]):
                voxel[c] = (voxel[c] - voxel[c].mean()) / voxel[c].std()

        return voxel

# Modify the get_datasets function to handle 3D data
def get_datasets(data_flag, download, as_rgb, resize, model_flag, size=224, shape_transform=False):
    # Added size and shape_transform parameters when integrating 3D datasets size changed back to 28 for 3d after taht change bavk to 224 
    info = INFO[data_flag]
    n_channels = 3 if as_rgb else info['n_channels']
    DataClass = getattr(medmnist, info['python_class'])

    if "3d" in data_flag or model_flag == 'vit_3d':
            transform = Transform3D()  # Use the updated Transform3DTo2D
            if as_rgb:
                transform = transforms.Compose([ToRGB(), transform])
            train_dataset = DataClass(split='train', transform=transform, download=download, as_rgb=as_rgb, size=size)
            val_dataset = DataClass(split='val', transform=transform, download=download, as_rgb=as_rgb, size=size)
            test_dataset = DataClass(split='test', transform=transform, download=download, as_rgb=as_rgb, size=size)
    else:
        # For 2D datasets
        transform_list = []

        if model_flag == 'resnet50': 
            # if model_flag == 'medclip_vit': used to be thid
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
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        transform = transforms.Compose(transform_list)

        train_dataset = DataClass(split='train', transform=transform, download=download, size=224)
        val_dataset = DataClass(split='val', transform=transform, download=download, size=224)
        test_dataset = DataClass(split='test', transform=transform, download=download, size=224)

    print("Datasets created:")
    print(f"Train dataset: {train_dataset is not None}")
    print(f"Val dataset: {val_dataset is not None}")
    print(f"Test dataset: {test_dataset is not None}")

    return train_dataset, val_dataset, test_dataset

# Existing get_dataloaders function remains unchanged
def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers, sampler=None):
    # Use the provided sampler if available; otherwise, default to shuffling
    if sampler:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
   
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

class ToRGB:
    def __call__(self, img):
        return img.convert("RGB")

class ToGrayscale:
    def __call__(self, img):
        return img.convert("L")

# Define the Transform3D class for 3D data
class Transform3D:
    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):
        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()
        return voxel.astype(np.float32)
    
#this is for 3d vit
class Transform3DTo2D:
    def __init__(self, resize_dim=(224, 224), mul=None):
        self.resize_dim = resize_dim
        self.mul = mul

    def __call__(self, voxel):
        # Apply any multiplication transformation
        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()

        # Assuming the voxel is of shape [D, H, W], we select slices along D or perform max pooling
        # Here, taking the middle slice along the depth dimension (D)
        middle_slice = voxel[voxel.shape[0] // 2, :, :]

        # Resize the slice to match the input size for ViT
        slice_resized = np.array(Image.fromarray(middle_slice).resize(self.resize_dim))

        # Normalize and convert to tensor
        slice_normalized = (slice_resized - np.mean(slice_resized)) / np.std(slice_resized)
        slice_tensor = torch.tensor(slice_normalized, dtype=torch.float32)

        # Add channel dimension since ViT expects [C, H, W]
        slice_tensor = slice_tensor.unsqueeze(0)

        return slice_tensor

# Define the function to convert models to synchronized batch normalization
# def model_to_syncbn(model):
#     preserve_state_dict = model.state_dict()  # Preserve the state before conversion
#     _convert_module_from_bn_to_syncbn(model)  # Convert to synchronized batch normalization
#     model.load_state_dict(preserve_state_dict)  # Reload the preserved state
#     return model

# def _convert_module_from_bn_to_syncbn(module):
#     for child_name, child in module.named_children():
#         if hasattr(child, 'num_features') and 'BatchNorm' in type(child).__name__:
#             # Replace with synchronized batch normalization
#             TargetClass = globals()[f'Synchronized{type(child).__name__}']
#             new_child = TargetClass(child.num_features, child.eps, child.momentum, child.affine)
#             setattr(module, child_name, new_child)
#         else:
#             _convert_module_from_bn_to_syncbn(child)


# Modify the get_datasets function to handle 3D data
def get_datasets(data_flag, download, as_rgb, resize, model_flag, size=28, shape_transform=False):
    # Added size and shape_transform parameters when integrating 3D datasets
    info = INFO[data_flag]
    n_channels = 3 if as_rgb else info['n_channels']
    DataClass = getattr(medmnist, info['python_class'])

    if "3d" in data_flag:
            transform = Transform3DTo2D(resize_dim=(224, 224))
            train_dataset = DataClass(split='train', transform=transform, download=download, as_rgb=as_rgb, size=size)
            val_dataset = DataClass(split='val', transform=transform, download=download, as_rgb=as_rgb, size=size)
            test_dataset = DataClass(split='test', transform=transform, download=download, as_rgb=as_rgb, size=size)
        #that was for resnet 18 
        # if "3d" in data_flag:  
        #     if shape_transform:
        #         train_transform = Transform3D(mul='random')
        #         eval_transform = Transform3D(mul='0.5')
        #     else:
        #         train_transform = eval_transform = Transform3D()
        #     # added as_rgb to true to perfectly align with benchmark to reproduce resulyt for resnet18 + ACS
        #     train_dataset = DataClass(split='train', transform=train_transform, download=download, as_rgb=True, size=size)
        #     val_dataset = DataClass(split='val', transform=eval_transform, download=download, as_rgb=True, size=size)
        #     test_dataset = DataClass(split='test', transform=eval_transform, download=download, as_rgb=True, size=size)
        
    else:
        # For 2D datasets
        transform_list = []

        if model_flag == 'medclip_vit':
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

        train_dataset = DataClass(split='train', transform=transform, download=download)
        val_dataset = DataClass(split='val', transform=transform, download=download)
        test_dataset = DataClass(split='test', transform=transform, download=download)

    print("Datasets created:")
    print(f"Train dataset: {train_dataset is not None}")
    print(f"Val dataset: {val_dataset is not None}")
    print(f"Test dataset: {test_dataset is not None}")

    return train_dataset, val_dataset, test_dataset

# Existing get_dataloaders function remains unchanged
def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

class ToRGB:
    def __call__(self, img):
        return img.repeat(3, 1, 1)

def get_datasets(data_flag, download, as_rgb, resize):
    info = INFO[data_flag]
    n_channels = 3 if as_rgb else info['n_channels']

    DataClass = getattr(medmnist, info['python_class'])

    transform_list = []
    if resize:
        transform_list.append(transforms.Resize((224, 224)))
    transform_list.append(transforms.ToTensor())
    if as_rgb and info['n_channels'] == 1:
        transform_list.append(ToRGB())
    # Normalize using ImageNet means and stds
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

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

#for fine tuning vit only those if train_dataset else none types removed idk why 
# def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset else None
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None
#     return train_loader, val_loader, test_loader



















# import medmnist
# from medmnist import INFO
# from torchvision import transforms
# from torch.utils.data import DataLoader
# import numpy as np

# class ToRGB:
#     def __call__(self, img):
#         return img.repeat(3, 1, 1)

# def get_datasets(data_flag, download, as_rgb, resize):
#     info = INFO[data_flag]
#     n_channels = 3 if as_rgb else info['n_channels']

#     DataClass = getattr(medmnist, info['python_class'])

#     transform_list = []
#     if resize:
#         transform_list.append(transforms.Resize((224, 224)))
#     transform_list.append(transforms.ToTensor())
#     if as_rgb and info['n_channels'] == 1:
#         transform_list.append(ToRGB())
#     transform_list.append(transforms.Normalize(mean=[.5], std=[.5]))

#     transform = transforms.Compose(transform_list)

#     train_dataset = DataClass(split='train', transform=transform, download=download)
#     val_dataset = DataClass(split='val', transform=transform, download=download)
#     test_dataset = DataClass(split='test', transform=transform, download=download)

#     print("Datasets created:")
#     print(f"Train dataset: {train_dataset is not None}")
#     print(f"Val dataset: {val_dataset is not None}")
#     print(f"Test dataset: {test_dataset is not None}")

#     return train_dataset, val_dataset, test_dataset

# def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset else None
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None
#     return train_loader, val_loader, test_loader

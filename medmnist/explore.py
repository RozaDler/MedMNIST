import medmnist
from medmnist import INFO
from medmnist import PathMNIST, PneumoniaMNIST, VesselMNIST3D
import matplotlib.pyplot as plt

"""This is for exploration 
1. loading the PathMNIST and PneumoniaMNIST datasets and exploring their properties
2. Visualizing images from the datasets.
3. understanding dataset splits: looking into train, validation, and test splits and sizes
4. checking labels of each class dataset 
"""
def load_dataset(dataset_class, split='train'):
    dataset = dataset_class(split=split, download=True)
    return dataset

def explore_dataset(dataset_class, dataset_name):
    print(f"Exploring {dataset_name}.....")
    dataset = load_dataset(dataset_class, 'train')
    print(dataset)

    #check the shape of the images
    print("Shape of images: ", dataset.imgs.shape)
    print("Shape of labels: ", dataset.labels.shape)

    #see some images 
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(dataset.imgs[i], cmap='gray' if dataset.info['n_channels'] == 1 else None)
        ax.set_title(f"Label: {dataset.labels[i]}")
        ax.axis('off')
    plt.show()

# PathMNIST 
explore_dataset(PathMNIST, "PathMNIST")

# PneumoniaMNIST
explore_dataset(PneumoniaMNIST, "PneumoniaMNIST")

# VesselMNIST3D
def explore_3d_dataset(dataset_class, dataset_name):
    print(f"Exploring {dataset_name} ...")
    dataset = load_dataset(dataset_class, 'train')
    print(dataset)

    # Check the shape of the images
    print("Shape of images:", dataset.imgs.shape)
    print("Shape of labels:", dataset.labels.shape)
    
    # Visualize some images
    fig, axes = plt.subplots(1, 3, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(dataset.imgs[i, 14], cmap='gray' if dataset.info['n_channels'] == 1 else None)
        ax.set_title(f"Label: {dataset.labels[i]}")
        ax.axis('off')
    plt.show()

explore_3d_dataset(VesselMNIST3D, "VesselMNIST3D")
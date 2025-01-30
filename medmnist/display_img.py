import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
from PIL import Image  # For loading the local image

# Function to get 3D datasets
def get_datasets_3d(data_flag, split='train', download=True):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load the 3D dataset
    dataset = DataClass(split=split, transform=None, download=download, size=64)
    
    return dataset

# Get the 3D dataset (VesselMNIST)
dataset_3d = get_datasets_3d('vesselmnist3d')

# Use the montage method to get 2D slices
frames = dataset_3d.montage(length=18)  # Generate the montage of frames

# Ensure you're selecting a single frame for display (e.g., the first frame in the montage)
frame_to_display = frames[0]  # Select the first frame to display (you can change the index)

# Load the local image of the vessel
local_image_path = "vessel.png"  # Replace with the path to your local image
local_image = Image.open(local_image_path)

# Plotting the montage
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(frame_to_display, cmap='gray')  # Display the first frame of the montage
ax.set_title('VesselMNIST 3D Montage')
ax.axis('off')  # Turn off axis for the montage

# Overlay the local image on top of the montage
zoom_inset = fig.add_axes([0.6, 0.6, 0.25, 0.25])  # Adjust position and size of the inset
zoom_inset.imshow(local_image, cmap='gray')  # Show the local image in the inset
zoom_inset.axis('off')  # Turn off axis for the local image

plt.show()
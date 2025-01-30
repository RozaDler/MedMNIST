import matplotlib.pyplot as plt
import numpy as np

# Data for the CNN curve (concave curve)
x_cnn = np.linspace(0, 20, 100)
y_cnn = 120 * (1 - np.exp(-0.3 * x_cnn))  # Concave curve representing CNN attention growth

# Sample ViT attention distances (approximation based on the provided image)
x_vit = np.arange(21)
y_vit = [20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 115, 118, 120, 120, 120, 120, 120, 120, 120]

# Plotting the ViT attention distances
fig, ax = plt.subplots()
ax.scatter(x_vit, y_vit, color='b', label='ViT Attention Distance')

# Adding the CNN curve
ax.plot(x_cnn, y_cnn, color='r', linestyle='--', label='CNN Attention Distance')

# Adding labels and title
ax.set_xlabel('Network Depth (Layer)')
ax.set_ylabel('Mean Attention Distance (pixels)')
ax.set_title('Attention Distance: CNN vs. ViT')
ax.legend()

# Display the plot
plt.show()





















# import numpy as np  # Add this import to fix the error
# import medmnist
# from medmnist import INFO
# import matplotlib.pyplot as plt

# # Function to get datasets without transformations
# def get_datasets(data_flag, split='test', download=True):
#     info = INFO[data_flag]
#     DataClass = getattr(medmnist, info['python_class'])

#     # Load the dataset without transformations
#     dataset = DataClass(split=split, transform=None, download=download)

#     return dataset

# # Example for PneumoniaMNIST (Binary Classification) - Pie Chart
# dataset = get_datasets('vesselmnist3d')

# # Get the labels and count their occurrences
# labels = dataset.labels
# unique, counts = np.unique(labels, return_counts=True)

# # Plot Pie Chart for Binary Classification
# plt.figure(figsize=(6,6))
# plt.pie(counts, labels=['Vessel', 'Aneurysm'], autopct='%1.1f%%', colors=['thistle', 'mediumpurple'])
# plt.title('VesselMNIST3D - Class Distribution (Test Set)')
# plt.show()

# # Example for PathMNIST (Multi-class Classification) - Bar Chart
# dataset = get_datasets('pathmnist')

# # Get the labels and count their occurrences
# labels = dataset.labels
# unique, counts = np.unique(labels, return_counts=True)

# # Get the class names for PathMNIST (this is a dictionary with string keys)
# class_names = INFO['pathmnist']['label']

# # Adjust indices to use string keys for dictionary lookup
# class_labels = [class_names[str(i)] for i in unique]

# # Plot Bar Chart for Multi-class Classification with thinner bars
# plt.figure(figsize=(12, 6))
# bars = plt.bar(range(len(unique)), counts, color='mediumpurple', width=0.8)  # Width set to 0.6 for thinner bars

# # Optional: Stacking long labels onto new lines
# def wrap_labels(labels, max_width=10):
#     wrapped_labels = []
#     for label in labels:
#         if len(label) > max_width:
#             wrapped_labels.append('\n'.join(label.split()))
#         else:
#             wrapped_labels.append(label)
#     return wrapped_labels

# # Apply the wrapping function to the class labels
# wrapped_class_labels = wrap_labels(class_labels)
# plt.xticks(range(len(unique)), wrapped_class_labels, rotation=0, ha='center', fontsize=10)

# # Set titles and labels
# plt.title('PathMNIST - Class Distribution (Training Set)')
# plt.xlabel('Class')
# plt.ylabel('Number of Samples')

# # Calculate percentages and add them as text on the bars
# total_samples = np.sum(counts)
# for i, count in enumerate(counts):
#     percentage = (count / total_samples) * 100
#     plt.text(i, count + 50, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, color='black')

# # Show the plot
# plt.show()
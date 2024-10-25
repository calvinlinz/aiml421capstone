import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict


# Define the classes
classes = ['cherry', 'strawberry', 'tomato']
data_dir = './testdata'

# Dictionary to store the loaded images
data = {}


for class_name in classes:
    images = []
    # Loop through all files in the class directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.jpg'):  # Check for image files
            file_path = os.path.join(class_dir, file_name)
            img = Image.open(file_path)
            images.append(img)
    # Store images for this class
    data[class_name] = images


# Example: Accessing images from the 'cherry' class
print(f'Loaded {len(data["cherry"])} images from cherry class.')
print(f'Loaded {len(data["strawberry"])} images from strawberry class.')
print(f'Loaded {len(data["tomato"])} images from tomato class.')


# Define the target resolution
target_size = (300, 300)

# Dictionary to hold filtered data
filtered_data = {}

count = 0
# Iterate through the classes
for class_name, images in data.items():
    filtered_images = []
    
    # Check each image for its resolution
    for img in images:
        if img.size == target_size:
            filtered_images.append(img)  # Keep images that match 300x300
        else:
            count += 1
    
    # Store only the filtered images in the new dictionary
    filtered_data[class_name] = filtered_images

# Example: Accessing filtered images
print(f'Filtered {len(filtered_data["cherry"])} images from cherry class.')
print(f'Filtered {len(filtered_data["strawberry"])} images from strawberry class.')
print(f'Filtered {len(filtered_data["tomato"])} images from tomato class.')
print(f'Removed {count} images in total.')
print(f'Filtered {len(filtered_data["cherry"])+len(filtered_data["strawberry"])+len(filtered_data["tomato"])} images in total.')


def detect_and_filter_rgb_outliers(image_data, thresholds):
    filtered_data = defaultdict(list)
    outliers = []
    grayscale_count = 0
    total_input_images = sum(len(images) for images in image_data.values())
    
    for class_name, images in image_data.items():
        for img in images:
            img_np = np.array(img)  # Convert image to NumPy array
            
            if len(img_np.shape) == 2:  # Grayscale image (only height and width)
                grayscale_count += 1
                continue
            
            # Calculate the mean pixel intensity for each RGB channel
            mean_channels = np.mean(img_np, axis=(0, 1))
            
            # Detect if any of the channels are outside their specific thresholds
            condition = (mean_channels < [t[0] for t in thresholds]) | (mean_channels > [t[1] for t in thresholds])
            if np.any(condition):
                outliers.append(img)
            else:
                filtered_data[class_name].append(img)
    
    total_processed_images = sum(len(images) for images in filtered_data.values()) + len(outliers)
    
    print(f"Input images: {total_input_images}")
    print(f"Processed images: {total_processed_images}")
    print(f"Removed Grayscale images: {grayscale_count}")
    print(f"RGB images: {total_processed_images - grayscale_count}")
    print(f"Outliers: {len(outliers)}")
    print(f"Images in filtered_data: {sum(len(images) for images in filtered_data.values())}")
    
    return dict(filtered_data), outliers

# Define channel-specific thresholds based on the distributions
thresholds = [
    (27, 238),  # Red channel (low, high)
    (14, 220),  # Green channel (low, high)
    (8, 218)    # Blue channel (low, high)
]

# Use the optimized function with new thresholds
filtered_data, rgb_outliers = detect_and_filter_rgb_outliers(filtered_data, thresholds)
print(f'\nFound {len(rgb_outliers)} potential RGB channel-based outliers out of {sum(len(images) for images in filtered_data.values()) + len(rgb_outliers)} total images.')
print(f'Filtered data contains {sum(len(images) for images in filtered_data.values())} images after RGB channel-based filtering.')

def normalize(data):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to tensor and normalizes to [0, 1]
    ])
    X_data = []
    y_labels = []

    label_mapping = {
        'cherry': 0,
        'strawberry': 1,
        'tomato': 2
    }

    # Step 1: Transform images directly without intermediate NumPy conversion
    for label, images in data.items():
        for img in images:
            img_transformed = transform(img)  # Apply transformation to normalize and convert to tensor
            X_data.append(img_transformed)
            y_labels.append(label_mapping[label])

    # Step 2: Stack tensors together
    X = torch.stack(X_data)  # Now, X will be of shape [num_images, 3, 300, 300]
    y = torch.tensor(y_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def get_dataloaders(batch_size):
    X_train, X_test, y_train, y_test = normalize(filtered_data)
    # Step 4: Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=64)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=64)
    # Check shapes
    print(f"Training X shape: {X_train.shape}, Training y shape: {y_train.shape}")
    print(f"Testing X shape: {X_test.shape}, Testing y shape: {y_test.shape}")
    return train_loader, test_loader


batch_size = 64
train_loader, test_loader = get_dataloaders(batch_size)


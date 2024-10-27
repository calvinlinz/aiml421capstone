from collections import defaultdict
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split, KFold
import random
import numpy as np
import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as F

classes = ['cherry', 'strawberry', 'tomato']
data_dir = 'testdata'
data = {}
target_size = (300, 300)
filtered_data = {}
count = 0

excluded_images = {
    'cherry_0055.jpg',
    'cherry_0105.jpg',
    'cherry_0147.jpg',
    'strawberry_0931.jpg',
    'tomato_0087.jpg'
}

for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    images = []
    
    for file_name in os.listdir(class_dir):
        if file_name.endswith('.jpg'): 
            if file_name in excluded_images:
                continue  
            file_path = os.path.join(class_dir, file_name)
            img = Image.open(file_path)
            images.append(img)
    
    # Store images for this class
    data[class_name] = images

print(f'Loaded {len(data["cherry"])} images from cherry class.')
print(f'Loaded {len(data["strawberry"])} images from strawberry class.')
print(f'Loaded {len(data["tomato"])} images from tomato class.')


def set_seed(seed):
    random.seed(seed)  # For Python random
    np.random.seed(seed)  # For NumPy random
    torch.manual_seed(seed)  # For PyTorch CPU random
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For PyTorch GPU random
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Make sure CUDA computations are deterministic
    torch.backends.cudnn.benchmark = False  # Disable benchmark mode to make it reproducible
set_seed(42)

for class_name, images in data.items():
    filtered_images = []
    for img in images:
        if img.size == target_size:
            filtered_images.append(img)  # Keep images that match 300x300
        else:
            count += 1
    
    # Store only the filtered images in the new dictionary
    filtered_data[class_name] = filtered_images

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
            mean_channels = np.mean(img_np, axis=(0, 1))
            condition = (mean_channels < [t[0] for t in thresholds]) | (mean_channels > [t[1] for t in thresholds])
            if np.any(condition):
                outliers.append(img)
            else:
                filtered_data[class_name].append(img)
    
    total_processed_images = sum(len(images) for images in filtered_data.values()) + len(outliers)
    
    print(f"Removed Grayscale images: {grayscale_count}")
    print(f"RGB images: {total_processed_images - grayscale_count}")
    print(f"Outliers: {len(outliers)}")
    print(f"Images in filtered_data: {sum(len(images) for images in filtered_data.values())}")
    return dict(filtered_data), outliers

thresholds = [
    (27, 238),  # Red channel (low, high)
    (14, 220),  # Green channel (low, high)
    (8, 218)    # Blue channel (low, high)
]

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

    for label, images in data.items():
        for img in images:
            img_transformed = transform(img)
            X_data.append(img_transformed)
            y_labels.append(label_mapping[label])

    X = torch.stack(X_data) 
    y = torch.tensor(y_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def get_dataloaders(batch_size):
    X_train, X_test, y_train, y_test = normalize(filtered_data)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    return train_loader, test_loader


batch_size = 64
train_loader, test_loader = get_dataloaders(batch_size)


class MLP(nn.Module):
    def __init__(self, input_size=300*300*3, hidden_size=512, output_size=3):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.flatten(x)  # Flatten the input image
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self):  # Use dropout_rate instead of decay for clarity
        super(CNN, self,).__init__()

        # First convolutional layer: input channels=3, output channels=16, kernel size=3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Max pooling layer to downsample
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 75 * 75, 128)  # Adjusting for 300x300 input size after pooling
        self.fc2 = nn.Linear(128, 3)  # Output size matches the number of classes (cherry, strawberry, tomato)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.0000147)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply first conv layer, activation, and pooling
        x = self.pool(self.relu(self.conv1(x)))

        # Apply second conv layer, activation, and pooling
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten the output from convolutional layers
        x = x.view(-1, 32 * 75 * 75)

        # Apply first fully connected layer with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout here

        # Apply second fully connected layer
        x = self.fc2(x)

        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss
    

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=3):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=-1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
    
    
    

def evaluate(model, val_loader, device ):
    model.eval()
    total, correct = 0, 0
    val_accuracies = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)
    return val_acc
    
def train_and_evaluate(train_loader, val_loader, model, optimizer, loss_function, device, num_epochs):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        total, correct = 0, 0
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        total, correct = 0, 0
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        # Store validation metrics
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training - Loss: {epoch_train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Validation - Loss: {epoch_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print("-" * 50)
    torch.save(model, 'model.pth')
    return val_acc


def k_fold(dataset, k_folds, trial=None):
    indices = list(range(len(dataset)))
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(device)    
        loss_function = LabelSmoothingLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.00020711569050274695, weight_decay=0.0001465691203881292)

        train_fold_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
        val_fold_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler)

        val_accuracy = train_and_evaluate(
            train_fold_loader, val_fold_loader,
            model, optimizer, loss_function,
            device, num_epochs=8
        )

        fold_accuracies.append(val_accuracy)

        if trial:
            trial.report(val_accuracy, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return fold_accuracies


def objective(trial):
    k_folds = 5
    dataset = train_loader.dataset
    fold_accuracies = k_fold(dataset, k_folds,trial)
    return np.mean(fold_accuracies)

def run_optimization(n_trials=10):
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=3600  # 1 hour timeout
    )
    
    print('\nBest trial:')
    print('Value:', study.best_value)
    print('Params:', study.best_params)
    
    return study

# study = run_optimization()

num_epochs = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
model.fc = nn.Linear(model.fc.in_features, 3).to(device)
loss_function = LabelSmoothingLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00020711569050274695)
result = train_and_evaluate(train_loader,test_loader,model,optimizer,loss_function,device,num_epochs)
print("Trained Model CV Results")
print(result)

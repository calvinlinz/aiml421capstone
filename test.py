
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data():
    print("Loading Data")
    classes = ['cherry', 'strawberry', 'tomato']
    data_dir = 'testdata'
    data = {}
    target_size = (300, 300)
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        images = []
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.jpg'): 
                file_path = os.path.join(class_dir, file_name)
                img = Image.open(file_path)
                img_np = np.array(img)
                if img.size != target_size or len(img_np.shape) == 2:
                    continue
                images.append(img)
        data[class_name] = images
        
    transform = transforms.Compose([
        transforms.ToTensor(), 
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
    print("Data Loaded")
    return X,y


def evaluate(model, val_loader, device ):
    print("Evaluating Model")
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    return val_acc


X,y = load_data()
tensor_dataset = TensorDataset(X, y)
data_loader = DataLoader(tensor_dataset, batch_size=64, shuffle=True, num_workers=8)
model = torch.load("model2.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Accuracy: " , evaluate(model,data_loader,device))
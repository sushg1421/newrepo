import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from medmnist import INFO, PathMNIST
from medmnist import Evaluator

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset info
data_flag = 'pathmnist'
info = INFO[data_flag]
num_classes = len(info['label'])

# Load dataset with transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = PathMNIST(split='train', transform=transform, download=True)
test_dataset = PathMNIST(split='test', transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet18 and adapt for MedMNIST
model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (1 epoch for demo)
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.squeeze().long().to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("✅ Training finished (1 epoch)")

# Evaluate on test set
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.squeeze().long().to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"🎯 Test Accuracy: {100 * correct / total:.2f}%")

# Show a few test images with predicted and actual labels
images, labels = next(iter(test_loader))
images = images.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

# Plot
class_names = ['background', 'normal', 'tumor', 'inflammatory', 'fibroblast', 'debris', 'mucus', 'immune cells', 'epithelial']
plt.figure(figsize=(12, 6))
for i in range(6):
    img = images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5  # un-normalize
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from medmnist import INFO, ChestMNIST

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load dataset
train_dataset = ChestMNIST(split='train', transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
info = INFO['chestmnist']
num_classes = len(info['label'])

# Define model
model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, num_classes),
    nn.Sigmoid()
)
model.to(device)

# Train
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(3):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).float()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 20 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Batch Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} complete | Total Loss: {total_loss:.4f}")


# Save model
torch.save(model.state_dict(), "chestmnist_pretrained.pth")
print("✅ Model saved as chestmnist_pretrained.pth")

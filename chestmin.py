import torchvision.transforms as transforms
from medmnist import INFO, ChestMNIST
from torch.utils.data import DataLoader

# Load dataset metadata
data_flag = 'chestmnist'
info = INFO[data_flag]

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load the dataset (downloads automatically if not present)
train_dataset = ChestMNIST(split='train', transform=transform, download=True)
test_dataset = ChestMNIST(split='test', transform=transform, download=True)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Sample output check
images, labels = next(iter(train_loader))
print(f"Image shape: {images.shape}")
print(f"Label shape: {labels.shape}")

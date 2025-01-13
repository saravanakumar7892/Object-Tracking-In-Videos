import sys
import os
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import LOGGER

# Add YOLOv5 directory to the path
sys.path.append('E:/Object_detection/yolov5')

# Configurations
dataset_path = "E:/Object_detection/datasets"  # Dataset location
weights_path = "E:/Object_detection/yolov5/runs/train/exp/weights/best.pt"  # Pretrained weights
batch_size = 64
num_epochs = 10
learning_rate = 0.001
img_size = 640
device = select_device('0')  # GPU/CPU

# Dataset class distribution (example distribution)
class_counts = [
    14256, 7881, 24825, 45612, 8705, 1479, 8871, 7297, 251, 2337, 581, 915, 122,
    1222, 8356, 5198, 631, 974, 1486, 1507, 28, 40, 20388, 48003, 8236, 335, 2082, 68, 384, 5926, 1080
]
total_samples = sum(class_counts)

# Calculate class weights and normalize
class_weights = [total_samples / count for count in class_counts]
weights = torch.tensor(class_weights, dtype=torch.float32)


# Custom Dataset Class
class CustomYOLODataset(Dataset):
    def __init__(self, dataset_path, img_size, transform=None):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.transform = transform
        self.image_files = []  # List of image file paths
        self.labels = []  # List of label arrays

        self._load_dataset()

    def _load_dataset(self):
        """Load image file paths and their corresponding labels."""
        image_dir = os.path.join(self.dataset_path, 'images')
        label_dir = os.path.join(self.dataset_path, 'labels')

        for image_file in os.listdir(image_dir):
            if image_file.endswith(('.jpg', '.png', '.jpeg')):
                label_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
                if os.path.exists(label_file):
                    self.image_files.append(os.path.join(image_dir, image_file))
                    with open(label_file, 'r') as f:
                        labels = [int(line.split()[0]) for line in f.readlines()]
                        self.labels.append(labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        labels = self.labels[idx]

        # Load and preprocess the image
        image = torch.load(image_path)  # Replace with an actual image loading method
        if self.transform:
            image = self.transform(image)

        return image, labels


# Dataset Transformations
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Load dataset
dataset = CustomYOLODataset(dataset_path, img_size, transform)

# Create sample weights for stratified sampling
sample_weights = torch.tensor([weights[label] for labels in dataset.labels for label in labels])
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# DataLoader
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)


# Load YOLOv5 model
model = attempt_load(weights_path, map_location=device)
model.to(device)

# Optimizer and Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()  # Optional: Mixed precision training for GPU

# Training Loop
LOGGER.info("Starting training...")
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=device.type != 'cpu'):
            pred = model(imgs)  # Predictions
            loss, loss_items = model.compute_loss(pred, targets)  # YOLOv5 custom loss

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track loss
        running_loss += loss.item()

        if batch_idx % 10 == 0:
            LOGGER.info(
                f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}"
            )

    # End of epoch
    epoch_loss = running_loss / len(train_loader)
    LOGGER.info(f"Epoch {epoch + 1} completed with Average Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "trained_model.pth")
LOGGER.info("Training complete. Model saved as 'trained_model.pth'.")

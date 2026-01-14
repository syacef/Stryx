import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from models.dataset import CCTDataset
from models.model import CustomResNet

# --- Configuration ---
# Path to the folder containing the actual images
DATA_ROOT = "/home/samy/dnn/cct_images/" 

# Path to the COCO Camera Traps .json file
ANNOTATIONS_FILE = "./caltech_images_20210113.json" 

# Your LightlySSL checkpoint
CHECKPOINT_PATH = "./out/ssl_baseline_long/checkpoints/last.ckpt" 

BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 1e-3

# --- Setup Data and Split ---
# 1. Pre-load JSON just to get all IDs for splitting
with open(ANNOTATIONS_FILE, 'r') as f:
    full_data = json.load(f)

# Extract IDs of images that actually have annotations
annotated_img_ids = set(ann['image_id'] for ann in full_data['annotations'])
all_valid_ids = list(annotated_img_ids)

# Split IDs into Train and Val
train_ids, val_ids = train_test_split(all_valid_ids, test_size=0.2, random_state=42)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create Datasets
train_dataset = CCTDataset(DATA_ROOT, ANNOTATIONS_FILE, transform=transform, image_ids=train_ids)
val_dataset = CCTDataset(DATA_ROOT, ANNOTATIONS_FILE, transform=transform, image_ids=val_ids)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize ResNet
resnet = models.resnet18()
# Remove fully connected layer to isolate backbone
backbone = nn.Sequential(*list(resnet.children())[:-1])

# Load Lightly Checkpoint
print(f"Loading SSL checkpoint from {CHECKPOINT_PATH}...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False    )
state_dict = checkpoint['state_dict']

# Clean state_dict keys (remove 'model.backbone.' or 'backbone.' prefixes)
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('model.backbone.'):
        new_state_dict[k.replace('model.backbone.', '')] = v
    elif k.startswith('backbone.'):
        new_state_dict[k.replace('backbone.', '')] = v
    # Sometimes Lightly saves the projection head too; we ignore that for classification

msg = backbone.load_state_dict(new_state_dict, strict=False)
print(f"Backbone loaded. Missing keys (expected for head): {msg.missing_keys}")

# Freeze Backbone (Linear Probe)
for param in backbone.parameters():
    param.requires_grad = False

model = CustomResNet(backbone, num_classes=len(train_dataset.classes)).to(device)

# --- Training Loop ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # Calculate metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * val_correct / val_total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Val Acc: {val_acc:.2f}%")

print("Training complete.")
# Save the final fine-tuned model
torch.save(model.state_dict(), "./out/cct_finetuned_resnet18.pth")
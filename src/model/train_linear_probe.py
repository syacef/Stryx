"""
Linear Probe Training Script

Trains a classifier on top of frozen SSL embeddings with proper validation
and early stopping to avoid overfitting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

from model.ssl.tinyvit import DistilledTinyViT
from dataset.supervised_dataset import SafariSupervisedDataset
from dataset.bbox_transforms import BboxAwareCrop, BboxAwareTransform
from compute_class_weights import compute_class_weights


class LinearProbeClassifier(nn.Module):
    """Simple classifier on top of SSL embeddings"""
    def __init__(self, embedding_dim=384, num_classes=91, use_refiner=True):
        super().__init__()
        self.use_refiner = use_refiner
        
        if use_refiner:
            # Use a refiner layer for better feature adaptation
            self.refiner = nn.Sequential(
                nn.Linear(embedding_dim, 512),
                nn.BatchNorm1d(512, track_running_stats=False),
                nn.SiLU(),
                nn.Dropout(0.3),
            )
            self.classifier = nn.Linear(512, num_classes)
        else:
            # Simple linear probe
            self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        if self.use_refiner:
            x = self.refiner(x)
        return self.classifier(x)


def extract_embeddings_batch(ssl_model, images, device):
    """Extract embeddings from images using SSL model"""
    ssl_model.eval()
    with torch.no_grad():
        images = images.to(device)
        # Handle 5D (video) or 4D (image) inputs
        if images.dim() == 5:
            b, t, c, h, w = images.shape
            images_reshaped = images.view(b * t, c, h, w)
            embeddings_all = ssl_model(images_reshaped)
            embeddings = embeddings_all.view(b, t, -1).mean(dim=1)
        else:
            embeddings = ssl_model(images)
    return embeddings


def train_epoch(ssl_model, classifier, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    classifier.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        labels = labels.to(device)
        
        # Extract embeddings with frozen SSL model
        embeddings = extract_embeddings_batch(ssl_model, images, device)
        
        # Forward pass through classifier
        optimizer.zero_grad()
        logits = classifier(embeddings)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def validate(ssl_model, classifier, val_loader, criterion, device):
    """Validate the model"""
    classifier.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            labels = labels.to(device)
            
            # Extract embeddings
            embeddings = extract_embeddings_batch(ssl_model, images, device)
            
            # Forward pass
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    return avg_loss, accuracy, f1


if __name__ == "__main__":
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SSL_CHECKPOINT = "weights/student_epoch24_tinyvit.pth"
    OUTPUT_PATH = "weights/linear_probe_best.pth"
    
    NUM_CLASSES = 91
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    PATIENCE = 10  # Early stopping patience
    USE_REFINER = True  # Use refiner layer for better performance
    
    print("=" * 80)
    print("LINEAR PROBE TRAINING")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"SSL Checkpoint: {SSL_CHECKPOINT}")
    print(f"Use Refiner: {USE_REFINER}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Epochs: {NUM_EPOCHS}")
    print(f"Early Stopping Patience: {PATIENCE}\n")
    
    # Load SSL model (frozen)
    print("[1/5] Loading SSL model...")
    ssl_model = DistilledTinyViT(teacher_dim=384).to(DEVICE)
    ssl_model.load_state_dict(torch.load(SSL_CHECKPOINT, map_location=DEVICE))
    ssl_model.eval()
    for param in ssl_model.parameters():
        param.requires_grad = False
    print("✓ SSL model loaded and frozen\n")
    
    # Prepare dataset with augmentation
    print("[2/5] Loading dataset...")
    
    # BboxAware training transform: 70% bbox-focused crop, 30% random
    bbox_crop = BboxAwareCrop(output_size=224, bbox_prob=0.7, margin_range=(1.2, 2.0))
    train_transform = BboxAwareTransform(
        bbox_crop=bbox_crop,
        additional_transforms=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    
    # Validation transform: standard center crop (no bbox awareness)
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create full dataset with bbox-aware training transform
    full_dataset = SafariSupervisedDataset(
        "./data/sa_fari_train_ext.json",
        "./data",
        "./data/labels.txt",
        transform=train_transform,
        bbox_json_path="src/model/data/annotated/train/sa_fari_train.json"
    )
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset_temp = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create validation dataset with different transform (standard center crop)
    val_dataset = SafariSupervisedDataset(
        "./data/sa_fari_train_ext.json",
        "./data",
        "./data/labels.txt",
        transform=val_transform,
        bbox_json_path="src/model/data/annotated/train/sa_fari_train.json"
    )
    val_dataset = torch.utils.data.Subset(val_dataset, val_dataset_temp.indices)
    
    # Load class weights for imbalanced dataset
    print("Loading class weights...")
    weights_file = "class_weights.json"
    
    if not os.path.exists(weights_file):
        print(f"⚠ {weights_file} not found. Computing class weights...")
        compute_class_weights(
            json_path="./data/sa_fari_train_ext.json",
            data_root="./data",
            output_file=weights_file
        )
        print(f"✓ Class weights computed and saved to {weights_file}\n")
    
    with open(weights_file, "r") as f:
        weights_data = json.load(f)
    
    # Create class weights tensor for loss function
    class_weights_dict = weights_data["class_weights"]
    # Convert string keys to int and create ordered tensor
    max_class_id = max(int(k) for k in class_weights_dict.keys())
    class_weights_tensor = torch.ones(max_class_id + 1)  # Default weight 1.0
    for class_id_str, weight in class_weights_dict.items():
        class_weights_tensor[int(class_id_str)] = weight
    class_weights_tensor = class_weights_tensor.to(DEVICE)
    
    # Compute sample weights for WeightedRandomSampler
    train_labels = [full_dataset.valid_samples[idx]["label"] for idx in train_dataset.indices]
    label_counts = Counter(train_labels)
    sample_weights = [1.0 / label_counts[label] for label in train_labels]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    
    # Create weighted sampler
    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=weighted_sampler,  # Use weighted sampler instead of shuffle
        num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    print(f"✓ Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print(f"✓ Loaded weights for {len(class_weights_dict)} classes")
    print(f"✓ Train set has {len(label_counts)} unique classes\n")
    
    # Create classifier
    print("[3/5] Creating classifier...")
    classifier = LinearProbeClassifier(
        embedding_dim=384,
        num_classes=NUM_CLASSES,
        use_refiner=USE_REFINER
    ).to(DEVICE)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"✓ Classifier created with {trainable_params:,} trainable parameters\n")
    
    # Training setup with weighted loss
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop with early stopping
    print("[4/5] Training...")
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            ssl_model, classifier, train_loader, optimizer, criterion, DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_f1 = validate(
            ssl_model, classifier, val_loader, criterion, DEVICE
        )
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), OUTPUT_PATH)
            print(f"✓ New best model saved! Val Acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("[5/5] TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {OUTPUT_PATH}")
    print("\nLoading best model for final evaluation...")
    
    classifier.load_state_dict(torch.load(OUTPUT_PATH, map_location=DEVICE))
    val_loss, val_acc, val_f1 = validate(
        ssl_model, classifier, val_loader, criterion, DEVICE
    )
    
    print(f"\nFinal Validation Results:")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  F1-Score: {val_f1:.4f}")

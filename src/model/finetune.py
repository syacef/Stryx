import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from dataset.supervised_dataset import SafariSupervisedDataset
from model.vit_student import SSLStudent
from model.safari_classifier import SafariMultiTaskClassifier

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct_species = 0
    total_samples = 0
    
    for features, species_targets, family_targets in loader:
        features, species_targets, family_targets = \
            features.to(device), species_targets.to(device), family_targets.to(device)

        # Forward
        species_logits, family_logits = model(features)
        
        # Loss Calculation
        loss_sp = F.cross_entropy(species_logits, species_targets)
        loss_fam = F.cross_entropy(family_logits, family_targets)
        loss = loss_sp + 0.4 * loss_fam

        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Accuracy Calculation
        _, predicted = torch.max(species_logits, 1)
        correct_species += (predicted == species_targets).sum().item()
        total_samples += species_targets.size(0)
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct_species / total_samples
    return avg_loss, accuracy

if __name__ == "__main__":
    device = torch.device("cuda")
    
    # Init
    base_student = SSLStudent(input_dim=384, embedding_dim=384)
    train_ds = SafariSupervisedDataset("./data/sa_fari_train_ext.json", "./data/embeddings")
    val_ds = SafariSupervisedDataset("./data/sa_fari_test_ext.json", "./data/embeddings")
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128)

    num_species = len(train_ds.species_map)
    num_families = len(train_ds.family_map)
    
    model = SafariMultiTaskClassifier(base_student, 384, num_species, num_families).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    
    # Scheduler
    epochs = 20
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=epochs)

    best_loss = float('inf')
    
    print(f"{'Epoch':<8} | {'Loss':<10} | {'Train Acc %':<12}")
    print("-" * 35)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        
        print(f"{epoch+1:<8} | {train_loss:<10.4f} | {train_acc:<12.2f}")

        # Save based on best loss since we are monitoring training performance
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "best_safari_model.pth")
    
    print(f"\nTraining complete. Best Train Loss: {best_loss:.4f}")

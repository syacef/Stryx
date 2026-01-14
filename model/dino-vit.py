import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
import os

# ==================== Dataset ====================
class CaltechCameraTrapsDataset(Dataset):
    def __init__(self, image_dir, processor, augment=True):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.rglob("*.jpg")) + list(self.image_dir.rglob("*.png"))
        self.processor = processor
        self.augment = augment
        
        # Strong augmentations for SSL
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.augment:
            # Create two augmented views of the same image (for contrastive learning)
            view1 = self.transform(image)
            view2 = self.transform(image)
            
            # Process with DINOv2 processor
            inputs1 = self.processor(images=view1, return_tensors="pt")
            inputs2 = self.processor(images=view2, return_tensors="pt")
            
            return {
                'pixel_values_1': inputs1['pixel_values'].squeeze(0),
                'pixel_values_2': inputs2['pixel_values'].squeeze(0)
            }
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            return {'pixel_values': inputs['pixel_values'].squeeze(0)}

# ==================== DINO Loss ====================
class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center = nn.Parameter(torch.zeros(1, out_dim))
        
    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of teacher and student networks
        """
        student_out = student_output / self.student_temp
        
        # Center + sharpen teacher output
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()
        
        student_out = F.log_softmax(student_out, dim=-1)
        
        loss = -torch.sum(teacher_out * student_out, dim=-1).mean()
        
        # Update center
        self.update_center(teacher_output)
        
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center.data = self.center.data * 0.9 + batch_center * 0.1

# ==================== DINO Head ====================
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

# ==================== Pretraining ====================
def pretrain_dino(
    image_dir,
    model_name="facebook/dinov2-base",
    output_dir="./pretrained_wildlife_dino",
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Load processor and model
    processor = AutoImageProcessor.from_pretrained(model_name)
    student_backbone = AutoModel.from_pretrained(model_name)
    teacher_backbone = AutoModel.from_pretrained(model_name)
    
    # Freeze teacher (it will be updated via EMA)
    for param in teacher_backbone.parameters():
        param.requires_grad = False
    
    # Get embedding dimension
    embed_dim = student_backbone.config.hidden_size  # 768 for base
    
    # Add projection heads
    student_head = DINOHead(embed_dim, out_dim=8192)
    teacher_head = DINOHead(embed_dim, out_dim=8192)
    teacher_head.load_state_dict(student_head.state_dict())
    
    # Move to device
    student_backbone = student_backbone.to(device)
    teacher_backbone = teacher_backbone.to(device)
    student_head = student_head.to(device)
    teacher_head = teacher_head.to(device)
    
    # Setup dataset and dataloader
    dataset = CaltechCameraTrapsDataset(image_dir, processor, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Loss and optimizer
    criterion = DINOLoss(out_dim=8192).to(device)
    optimizer = torch.optim.AdamW(
        list(student_backbone.parameters()) + list(student_head.parameters()),
        lr=learning_rate,
        weight_decay=0.04
    )
    
    # Cosine learning rate schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))
    
    print(f"Starting pretraining on {len(dataset)} images")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
    
    # Training loop
    for epoch in range(num_epochs):
        student_backbone.train()
        student_head.train()
        teacher_backbone.eval()
        teacher_head.eval()
        
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            view1 = batch['pixel_values_1'].to(device)
            view2 = batch['pixel_values_2'].to(device)
            
            # Student forward pass
            student_output1 = student_backbone(pixel_values=view1).last_hidden_state[:, 0]  # CLS token
            student_output2 = student_backbone(pixel_values=view2).last_hidden_state[:, 0]
            
            student_proj1 = student_head(student_output1)
            student_proj2 = student_head(student_output2)
            
            # Teacher forward pass (no gradient)
            with torch.no_grad():
                teacher_output1 = teacher_backbone(pixel_values=view1).last_hidden_state[:, 0]
                teacher_output2 = teacher_backbone(pixel_values=view2).last_hidden_state[:, 0]
                
                teacher_proj1 = teacher_head(teacher_output1)
                teacher_proj2 = teacher_head(teacher_output2)
            
            # DINO loss (cross-view prediction)
            loss = (criterion(student_proj1, teacher_proj2) + criterion(student_proj2, teacher_proj1)) / 2
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # EMA update for teacher
            with torch.no_grad():
                m = 0.996  # momentum parameter
                for param_student, param_teacher in zip(student_backbone.parameters(), teacher_backbone.parameters()):
                    param_teacher.data = param_teacher.data * m + param_student.data * (1 - m)
                for param_student, param_teacher in zip(student_head.parameters(), teacher_head.parameters()):
                    param_teacher.data = param_teacher.data * m + param_student.data * (1 - m)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = f"{output_dir}/dinov2_pretrained_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'student_backbone_state_dict': student_backbone.state_dict(),
                'student_head_state_dict': student_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print("Pretraining complete!")
    return student_backbone

# ==================== Usage ====================
if __name__ == "__main__":
    # Path to your Caltech Camera Traps images
    IMAGE_DIR = "/home/milik/Documents/scia/s9/dnn/project/data/"  # UPDATE THIS
    
    # Pretrain the model
    pretrained_model = pretrain_dino(
        image_dir=IMAGE_DIR,
        num_epochs=20,  # Adjust based on your needs
        batch_size=32,   # Adjust based on your GPU memory
        learning_rate=1e-4
    )
    
    # Here starts fine-tuning

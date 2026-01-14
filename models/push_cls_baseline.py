import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_tracker.project_tracker import WildlifeTracker
from models.model import CustomResNet
from torchvision import models
import torch
import torch.nn as nn

# 1. Setup
tracker = WildlifeTracker(task_type="cls")

# 2. Training Loop
with tracker.start_run(run_name="baseline-cls-long"):
    # 3. Create backbone matching fine_tune_classifier.py structure
    resnet = models.resnet18(pretrained=False)
    # Remove fully connected layer to isolate backbone
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    
    # Load SSL checkpoint into backbone
    ssl_checkpoint_path = "./out/cct_finetuned_resnet18_long.pth"
    print(f"Loading SSL checkpoint from {ssl_checkpoint_path}...")
    checkpoint = torch.load(ssl_checkpoint_path, weights_only=False)
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Checkpoint might be the state dict directly
            state_dict = checkpoint
    else:
        # Checkpoint is already the state dict
        state_dict = checkpoint
    
    # Clean state_dict keys (remove 'model.backbone.' or 'backbone.' prefixes)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.backbone.'):
            new_state_dict[k.replace('model.backbone.', '')] = v
        elif k.startswith('backbone.'):
            new_state_dict[k.replace('backbone.', '')] = v
    
    backbone.load_state_dict(new_state_dict, strict=False)
    
    
    # Create CustomResNet with the backbone
    num_classes = 22
    my_ssl_model = CustomResNet(backbone=backbone, num_classes=num_classes)
    
    finetuned_path = "../out/cct_finetuned_resnet18.pth"
    if os.path.exists(finetuned_path):
        print(f"Loading fine-tuned model from {finetuned_path}...")
        my_ssl_model.load_state_dict(torch.load(finetuned_path, weights_only=False))
    
    # 4. Save with one line
    tracker.log_classifier(my_ssl_model, accuracy=0.6417, source_ssl_version="5")
    
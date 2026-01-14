import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_tracker.project_tracker import WildlifeTracker
from torchvision import models
import torch
# 1. Setup
tracker = WildlifeTracker(task_type="ssl")

# 2. Training Loop
with tracker.start_run(run_name="dinov2_vitb14-batch64-epoch1"):
    # 3. Load your trained model here or train it from scratch
    my_ssl_model =  models.resnet18(pretrained=False)
    
    # Load Lightning checkpoint and extract the model state_dict
    checkpoint = torch.load("../out/ssl_baseline_1/checkpoints/last.ckpt", weights_only=False)
    model_state_dict = checkpoint["state_dict"]
    
    # Remove 'model.' prefix if present (Lightning often adds this)
    cleaned_state_dict = {k.replace('model.', ''): v for k, v in model_state_dict.items()}
    
    my_ssl_model.load_state_dict(cleaned_state_dict, strict=False)
    
    # 4. Save with one line
    tracker.log_ssl_model(my_ssl_model)
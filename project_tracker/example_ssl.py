from project_tracker import WildlifeTracker
from torchvision import models
import torch
# 1. Setup
tracker = WildlifeTracker(task_type="ssl")

# 2. Training Loop
with tracker.start_run(run_name="dinov2_vitb14-batch64-epoch1"):
    # 3. Load your trained model here or train it from scratch
    my_ssl_model =  models.resnet18(pretrained=False)

    my_ssl_model.load_state_dict(torch.load("../out/ssl_baseline_1/checkpoints/last.ckpt"), weights_only=False)
    
    # 4. Save with one line
    tracker.log_ssl_model(my_ssl_model)
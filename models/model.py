import torch.nn as nn

# Attach Classification Head
class CustomResNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
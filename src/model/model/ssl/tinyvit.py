import torch.nn as nn
import timm


class DistilledTinyViT(nn.Module):
    def __init__(self, teacher_dim=384):
        super().__init__()
        self.student = timm.create_model(
            "tiny_vit_5m_224.in1k", pretrained=False, num_classes=0
        )
        self.projector = nn.Linear(320, teacher_dim)

    def forward(self, x):
        feat = self.student(x)
        return self.projector(feat)

import torch.nn as nn


class SafariSpeciesClassifier(nn.Module):
    def __init__(self, backbone, embedding_dim, num_species):
        super().__init__()
        self.backbone = backbone

        # 1. Feature Projector (Refining the backbone output)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.2),
        )

        # 2. Species Head
        self.species_head = nn.Sequential(
            nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, num_species)
        )

    def forward(self, x):
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)

            features = self.backbone(x)
            features = features.view(b, t, -1)
            features = features.mean(dim=1)
        else:
            features = self.backbone(x)

        # Final classification
        x = self.proj(features)
        return self.species_head(x)

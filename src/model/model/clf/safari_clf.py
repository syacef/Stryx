import torch.nn as nn


class SafariSpeciesClassifier(nn.Module):
    def __init__(self, student_model, embedding_dim, num_species):
        super().__init__()
        self.backbone = student_model

        # Bottleneck / Refiner
        self.refiner = nn.Sequential(
            nn.Linear(embedding_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3)
        )

        self.species_head = nn.Linear(512, num_species)

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
        x = self.refiner(features)
        return self.species_head(x)

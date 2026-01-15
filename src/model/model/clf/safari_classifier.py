from torch import nn


class SafariMultiTaskClassifier(nn.Module):
    def __init__(self, student_model, embedding_dim, num_species, num_families):
        super().__init__()
        self.backbone = student_model

        # Bottleneck / Refiner
        self.refiner = nn.Sequential(
            nn.Linear(embedding_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3)
        )

        self.species_head = nn.Linear(512, num_species)
        self.family_head = nn.Linear(512, num_families)

    def forward(self, x):
        features = self.backbone(x)
        x = self.refiner(features)
        return self.species_head(x), self.family_head(x)

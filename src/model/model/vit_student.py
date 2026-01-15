import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x + self.block(x))

class SSLStudent(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=1024, embedding_dim=384, depth=3):
        super().__init__()
        # Initial expansion
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Deeper residual core
        self.core = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(depth)])
        
        # Final embedding projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            # No activation on the final layer for SSL embeddings
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.core(x)
        return self.output_proj(x)

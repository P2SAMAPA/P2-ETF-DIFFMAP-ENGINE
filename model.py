import torch
import torch.nn as nn

class DiffMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.context = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.sigma_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, sigma):
        c = self.context(x)
        s = self.sigma_embed(sigma)
        x = torch.cat([c, s], dim=1)
        return self.head(x)

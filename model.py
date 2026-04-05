import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """
    Encodes sigma/time into a frequency-based embedding.
    This helps the model 'see' different noise scales more clearly.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)
        self.activation = nn.SiLU()  # Swish activation is standard for modern Diffusion
        self.shortcut = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        return self.activation(self.norm(self.linear(x))) + self.shortcut(x)

class DiffMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 1. Advanced Time/Sigma Embedding (16-dim)
        self.sigma_mlp = nn.Sequential(
            SinusoidalPosEmb(16),
            nn.Linear(16, 32),
            nn.SiLU(),
            nn.Linear(32, 16)
        )

        # 2. Context Processing with Residuals
        self.context_layer1 = ResidualBlock(input_dim, 128)
        self.context_layer2 = ResidualBlock(128, 64)

        # 3. Prediction Head
        # input_dim 64 (context) + 16 (sigma) = 80
        self.head = nn.Sequential(
            nn.Linear(80, 64),
            nn.SiLU(),
            nn.Dropout(0.1),  # Helps prevent overfitting on historical noise
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, sigma):
        """
        x: ETF Feature tensor [batch, input_dim]
        sigma: Noise level [batch, 1]
        """
        # Embed sigma/time
        s_emb = self.sigma_mlp(sigma.flatten())
        
        # Process context
        c = self.context_layer1(x)
        c = self.context_layer2(c)
        
        # Concatenate and Predict
        combined = torch.cat([c, s_emb], dim=1)
        return self.head(combined)

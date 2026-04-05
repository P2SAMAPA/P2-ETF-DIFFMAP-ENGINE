import torch
import torch.nn as nn
import numpy as np
from model import DiffMLP
from config import MACRO_VARS, LOOKBACK, LR, EPOCHS, BATCH_SIZE, SIGMA_MIN, SIGMA_MAX

def train_model(df, etf):
    # Ensure we use a copy to avoid SettingWithCopy warnings in live pandas dfs
    features = [etf] + MACRO_VARS
    data = df[features].dropna().copy().values

    if len(data) < LOOKBACK + 1:
        raise ValueError(
            f"Insufficient data for {etf}: got {len(data)} rows, "
            f"need at least {LOOKBACK + 1}."
        )

    X_list, y_list = [], []
    # Logic: Use LOOKBACK days of features to predict the NEXT day's ETF return
    for i in range(LOOKBACK, len(data)):
        # Context: Previous days (flattened)
        X_list.append(data[i - LOOKBACK:i].flatten())
        # Target: Current day's ETF return (index 0)
        y_list.append(data[i, 0])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffMLP(X.shape[1]).to(device)
    X, y = X.to(device), y.to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # Exponential Moving Average (EMA) is recommended for stable live diffusion
    # but omitted here to keep the file "swap-in" ready without extra classes.

    # P2-Weighting Hyperparameters (k=1, gamma=0.5 are standard for P2)
    P2_K = 1.0
    P2_GAMMA = 0.5 

    model.train()
    for epoch in range(EPOCHS):
        # Shuffle batch indices for better convergence
        permutation = torch.randperm(len(X))
        
        for i in range(0, len(X), BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]
            xb, yb = X[indices], y[indices]
            
            # 1. Sample Noise Levels (Sigma)
            sigma = torch.rand(len(xb), 1, device=device) * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN
            
            # 2. Add Noise to Target (Forward Process)
            noise = torch.randn_like(yb)
            # x_sigma = target + noise * sigma
            x_sigma = yb + noise * sigma
            
            # 3. Predict Noise (The model estimates 'noise' given 'x_sigma' and 'xb')
            pred_noise = model(xb, sigma) # Note: we pass context 'xb' and noise level 'sigma'
            
            # 4. P2-Weighting Loss Calculation
            # Weight = 1 / (k + sigma^2)^gamma
            weight = 1.0 / (P2_K + sigma**2)**P2_GAMMA
            
            # Mean Squared Error weighted by P2 factor
            loss = (weight * (pred_noise - noise)**2).mean()
            
            opt.zero_grad()
            loss.backward()
            # Gradient clipping to prevent 'shitty' exploding gradients in live data
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return model

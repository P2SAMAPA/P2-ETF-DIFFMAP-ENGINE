import torch
import numpy as np
from model import DiffMLP
from config import MACRO_VARS, LOOKBACK, LR, EPOCHS, BATCH_SIZE, SIGMA_MIN, SIGMA_MAX

def train_model(df, etf):
    data = df[[etf] + MACRO_VARS].dropna().values

    if len(data) < LOOKBACK + 2:
        raise ValueError(
            f"Insufficient data for {etf}: got {len(data)} rows, "
            f"need at least {LOOKBACK + 2}. "
            f"Check your HuggingFace dataset has enough history."
        )

    X, y = [], []
    for i in range(LOOKBACK, len(data) - 1):
        X.append(data[i - LOOKBACK:i].flatten())
        y.append(data[i][0])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    model = DiffMLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i + BATCH_SIZE]
            yb = y[i:i + BATCH_SIZE]
            sigma = torch.rand(len(xb), 1) * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN
            noise = torch.randn_like(yb)
            x_sigma = yb + sigma * noise
            pred = model(xb, sigma)
            loss = ((pred - noise) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model

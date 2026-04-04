import torch
import numpy as np
from config import MACRO_VARS, LOOKBACK, SIGMA_MIN, SIGMA_MAX

def predict_etf(model, df, etf):
    """
    Returns:
        mu: expected return (mean over sigma samples)
        p_up: probability of positive return
    """
    model.eval()

    data = df[[etf] + MACRO_VARS].dropna().values
    if len(data) < LOOKBACK:
        return 0.0, 0.5

    context = torch.tensor(
        data[-LOOKBACK:].flatten(),
        dtype=torch.float32
    ).unsqueeze(0)

    # Average prediction over multiple sigma levels
    # model was trained with random sigma — sample several at inference
    n_sigma = 20
    preds = []
    with torch.no_grad():
        for _ in range(n_sigma):
            sigma_val = np.random.uniform(SIGMA_MIN, SIGMA_MAX)
            sigma = torch.tensor([[sigma_val]], dtype=torch.float32)
            pred = model(context, sigma)
            preds.append(pred.item())

    mu = float(np.mean(preds))
    p_up = float(np.mean([p > 0 for p in preds]))

    return mu, p_up

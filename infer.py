# infer.py

import torch
import numpy as np

from config import MACRO_VARS, LOOKBACK


def predict_etf(model, df, etf):
    """
    Returns:
        mu: expected return
        p_up: probability of positive return
    """

    model.eval()

    # prepare input
    data = df[[etf] + MACRO_VARS].dropna().values

    if len(data) < LOOKBACK:
        return 0.0, 0.5

    context = torch.tensor(
        data[-LOOKBACK:].flatten(),
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        preds = model(context)

    # assume model outputs mean return
    mu = preds.item()

    # simple probability proxy (can improve later)
    p_up = float(mu > 0)

    return float(mu), p_up

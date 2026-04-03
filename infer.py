import numpy as np
from config import *
from sampler import sample_returns

def predict_etf(model, df, etf):

    data = df[[etf] + MACRO_VARS].dropna().values
    context = data[-LOOKBACK:].flatten()

    import torch
    context = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

    samples = sample_returns(model, context, N_SAMPLES)

    mu = samples.mean()
    p_up = (samples > 0).mean()

    return mu, p_up

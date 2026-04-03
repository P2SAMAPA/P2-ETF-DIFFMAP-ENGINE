import numpy as np
from config import *
from sampler import sample_returns

def predict_all(model_dict, df):

    results = {}
    samples_store = {}

    import torch

    for etf in ALL_ETFS:
        data = df[[etf] + MACRO_VARS].dropna().values
        context = data[-LOOKBACK:].flatten()
        context = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        samples = sample_returns(model_dict["A"], context, N_SAMPLES)

        mu = samples.mean()

        results[etf] = mu
        samples_store[etf] = samples

    return results, samples_store

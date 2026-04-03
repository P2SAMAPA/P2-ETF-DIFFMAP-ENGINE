import torch
import numpy as np
from config import SIGMA_MIN, SIGMA_MAX

def sample_returns(model, context, n_samples):
    model.eval()
    samples = []

    for _ in range(n_samples):
        sigma = np.random.uniform(SIGMA_MIN, SIGMA_MAX)
        z = np.random.randn()

        x_sigma = z * sigma

        with torch.no_grad():
            eps = model(context, torch.tensor([[sigma]], dtype=torch.float32))
        
        x0 = x_sigma - sigma * eps.item()
        samples.append(x0)

    return np.array(samples)

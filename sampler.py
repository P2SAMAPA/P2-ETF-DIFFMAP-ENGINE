import math
import torch
import numpy as np
from config import SIGMA_MIN, SIGMA_MAX

def sample_returns(model, context, n_samples, steps=30):
    """
    Implements a proper Iterative Reverse Diffusion sampler.
    Uses a deterministic ODE solver (Heun's method) for better consistency.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Ensure context is the right shape [1, input_dim]
    if context.ndim == 1:
        context = context.unsqueeze(0)
    context = context.to(device)

    # 1. Create a log-spaced noise schedule (Standard for EDM/DiffMap)
    # We start at SIGMA_MAX (pure noise) and go to SIGMA_MIN (clean data)
    t_steps = torch.linspace(math.log(SIGMA_MAX), math.log(SIGMA_MIN), steps, device=device).exp()
    # Add a zero at the end for the final denoising step
    t_steps = torch.cat([t_steps, torch.zeros(1, device=device)])

    # 2. Initialize with pure Gaussian noise
    # We generate n_samples in parallel for efficiency
    x_next = torch.randn((n_samples, 1), device=device) * t_steps[0]
    
    with torch.no_grad():
        for i in range(steps):
            t_cur = t_steps[i]
            t_next = t_steps[i+1]
            
            # Expand t_cur to match batch size
            t_tensor = torch.full((n_samples, 1), t_cur, device=device)

            # --- Heun's Method (2nd order ODE) ---
            # 1. Predict noise at current level
            d_cur = model(context.repeat(n_samples, 1), t_tensor)
            
            # Euler step
            x_prime = x_next + (t_next - t_cur) * d_cur
            
            # 2. Correction step (if not at the last step)
            if t_next > 0:
                t_next_tensor = torch.full((n_samples, 1), t_next, device=device)
                d_prime = model(context.repeat(n_samples, 1), t_next_tensor)
                x_next = x_next + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
            else:
                x_next = x_prime

    return x_next.cpu().numpy().flatten()

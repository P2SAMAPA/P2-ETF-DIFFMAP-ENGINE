# run.py

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from data_loader import load_data
from train import train_model
from infer import predict_etf
from config import WINDOWS, ALL_ETFS, MACRO_VARS, LOOKBACK
from portfolio import PortfolioState
from utils import compute_tbill_daily_rate
from calendar_utils import get_next_trading_day

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = load_data()
df = df.sort_values("date")

# ─────────────────────────────────────────────
# TRAIN MODELS PER WINDOW
# ─────────────────────────────────────────────
models = {}

for w, start in WINDOWS.items():
    df_w = df[df["date"] >= start]
    models[w] = {}

    for etf in ALL_ETFS:
        models[w][etf] = train_model(df_w, etf)

# ─────────────────────────────────────────────
# PREDICTIONS (PER WINDOW, PER ETF)
# ─────────────────────────────────────────────
window_preds = {}
window_samples = {}

for w in models:
    window_preds[w] = {}
    window_samples[w] = {}

    for etf in ALL_ETFS:
        mu, p_up = predict_etf(models[w][etf], df, etf)

        window_preds[w][etf] = mu

        # regenerate samples for UI + agreement
        from sampler import sample_returns
        import torch

        data = df[[etf] + MACRO_VARS].dropna().values
        context = torch.tensor(
            data[-LOOKBACK:].flatten(),
            dtype=torch.float32
        ).unsqueeze(0)

        samples = sample_returns(models[w][etf], context, 100)

        window_samples[w][etf] = samples

# ─────────────────────────────────────────────
# AGGREGATE ACROSS WINDOWS
# ─────────────────────────────────────────────
final_scores = {}
agreement = {}
all_samples = {}

for etf in ALL_ETFS:
    mus = []
    pos_count = 0
    combined_samples = []

    for w in WINDOWS:
        mu = window_preds[w][etf]
        mus.append(mu)

        samples = window_samples[w][etf]
        combined_samples.extend(samples)

        if mu > 0:
            pos_count += 1

    final_scores[etf] = np.mean(mus)
    agreement[etf] = pos_count
    all_samples[etf] = np.array(combined_samples)

# ─────────────────────────────────────────────
# TOP 3 ETFs
# ─────────────────────────────────────────────
sorted_etfs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

top3 = [
    {"etf": e, "mu": float(m)}
    for e, m in sorted_etfs[:3]
]

# ─────────────────────────────────────────────
# PORTFOLIO DECISION
# ─────────────────────────────────────────────
portfolio = PortfolioState()
tbill = compute_tbill_daily_rate(df)

pick, score = portfolio.decide(final_scores.copy(), all_samples, tbill)

# mode detection
mode = "NORMAL"
if portfolio.in_cash:
    mode = "CASH"
if portfolio.check_tsl():
    mode = "TSL_ACTIVE"

# confidence
samples_best = all_samples.get(pick, np.array([0]))
confidence = float((samples_best > 0).mean())

# ─────────────────────────────────────────────
# REAL BACKTEST (ROLLING, NO LEAKAGE)
# ─────────────────────────────────────────────
equity = [1.0]
portfolio_bt = PortfolioState()

returns_df = df.set_index("date")[ALL_ETFS]

for i in range(LOOKBACK, len(returns_df)-1):

    sub_df = df.iloc[:i]

    preds_bt = {}
    samples_bt = {}

    for etf in ALL_ETFS:
        model_bt = train_model(sub_df, etf)
        mu, _ = predict_etf(model_bt, sub_df, etf)

        preds_bt[etf] = mu

        # simple sample proxy
        samples_bt[etf] = np.random.normal(mu, 0.01, 50)

    tbill_bt = compute_tbill_daily_rate(sub_df)

    pick_bt, _ = portfolio_bt.decide(preds_bt, samples_bt, tbill_bt)

    if pick_bt == "CASH":
        r = tbill_bt
    else:
        r = returns_df.iloc[i+1][pick_bt]

    portfolio_bt.update_returns(r)

    equity.append(equity[-1] * (1 + r))

equity_curve = equity[-250:]

# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────
output = {
    "engine": "DIFFMAP-ETF",
    "date": datetime.utcnow().strftime("%Y-%m-%d"),
    "next_trading_day": get_next_trading_day(),
    "pick": pick,
    "score": float(score),
    "confidence": confidence,
    "mode": mode,
    "top_3": top3,
    "window_scores": {
        w: float(window_preds[w][pick]) for w in WINDOWS
    },
    "samples": {
        k: v.tolist() for k, v in all_samples.items()
    },
    "equity_curve": equity_curve,
    "agreement": agreement
}

os.makedirs("outputs", exist_ok=True)

fname = f"outputs/diffmap_{output['date']}.json"

with open(fname, "w") as f:
    json.dump(output, f, indent=2)

print(output)

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from huggingface_hub import HfApi

from data_loader import load_data
from train import train_model
from infer import predict_etf
from sampler import sample_returns
from config import WINDOWS, ALL_ETFS, MACRO_VARS, LOOKBACK, HF_OUTPUT_DATASET
from portfolio import PortfolioState
from utils import compute_tbill_daily_rate
from calendar_utils import get_next_trading_day

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = load_data()
df = df.sort_values("date")

if df.empty:
    raise ValueError("Loaded dataframe is empty. Check data source.")

print(f"Data loaded: {len(df)} rows from {df['date'].min()} to {df['date'].max()}")

df = df.set_index("date")
df = df.sort_index()

# ─────────────────────────────────────────────
# TRAIN MODELS PER WINDOW + ETF
# ─────────────────────────────────────────────
models = {}

for w, start in WINDOWS.items():
    df_w = df[df.index >= start]

    if df_w.empty:
        print(f"Warning: No data for window {w} starting {start}")
        continue

    models[w] = {}

    for etf in ALL_ETFS:
        if etf in df_w.columns:
            models[w][etf] = train_model(df_w, etf)
        else:
            print(f"Warning: ETF {etf} not found in data for window {w}")

# ─────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────
window_preds = {}
window_samples = {}

import torch

for w in models:
    window_preds[w] = {}
    window_samples[w] = {}

    for etf in ALL_ETFS:
        if etf not in models[w]:
            continue

        mu, _ = predict_etf(models[w][etf], df.reset_index(), etf)
        window_preds[w][etf] = mu

        data_df = df[[etf] + MACRO_VARS].dropna()
        if len(data_df) < LOOKBACK:
            print(f"Warning: Insufficient data for {etf} in window {w}")
            continue

        data = data_df.values
        context = torch.tensor(
            data[-LOOKBACK:].flatten(),
            dtype=torch.float32
        ).unsqueeze(0)

        samples = sample_returns(models[w][etf], context, 100)
        window_samples[w][etf] = samples

# ─────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────
final_scores = {}
agreement = {}
all_samples = {}

for etf in ALL_ETFS:
    mus = []
    pos = 0
    combined = []

    for w in WINDOWS:
        if w not in window_preds:
            continue
        if etf not in window_preds[w]:
            continue

        mu = window_preds[w][etf]
        mus.append(mu)

        if w in window_samples and etf in window_samples[w]:
            s = window_samples[w][etf]
            combined.extend(s)

        if mu > 0:
            pos += 1

    if mus:
        final_scores[etf] = np.mean(mus)
        agreement[etf] = pos
        all_samples[etf] = np.array(combined) if combined else np.array([0])
    else:
        final_scores[etf] = 0.0
        agreement[etf] = 0
        all_samples[etf] = np.array([0])

# ─────────────────────────────────────────────
# TOP 3
# ─────────────────────────────────────────────
sorted_etfs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
top3 = [{"etf": e, "mu": float(m)} for e, m in sorted_etfs[:3]]

# ─────────────────────────────────────────────
# PORTFOLIO DECISION
# ─────────────────────────────────────────────
portfolio = PortfolioState()
tbill = compute_tbill_daily_rate(df.reset_index())

pick, score = portfolio.decide(final_scores.copy(), all_samples, tbill)

mode = "NORMAL"
if portfolio.in_cash:
    mode = "CASH"
if portfolio.check_tsl():
    mode = "TSL_ACTIVE"

samples_best = all_samples.get(pick, np.array([0]))
confidence = float((samples_best > 0).mean()) if len(samples_best) > 0 else 0.0

# ─────────────────────────────────────────────
# LIGHT BACKTEST (FAST)
# ─────────────────────────────────────────────
equity = [1.0]

# Data is already returns — do NOT call pct_change() again
returns_df = df[ALL_ETFS].dropna()

portfolio_bt = PortfolioState()
backtest_returns = []

for i in range(LOOKBACK, len(returns_df) - 1):
    row = returns_df.iloc[i]
    next_row = returns_df.iloc[i + 1]

    preds_bt = row.to_dict()

    samples_bt = {}
    for k, v in preds_bt.items():
        hist_vol = returns_df[k].rolling(20).std().iloc[i] if i >= 20 else 0.01
        samples_bt[k] = np.random.normal(v, abs(hist_vol) + 0.001, 50)

    tbill_bt = compute_tbill_daily_rate(df.reset_index().iloc[:i + LOOKBACK])

    pick_bt, _ = portfolio_bt.decide(preds_bt, samples_bt, tbill_bt)

    if pick_bt == "CASH":
        r = tbill_bt
    else:
        r = next_row[pick_bt] if pick_bt in next_row.index else 0.0

    portfolio_bt.update_returns(r)
    backtest_returns.append(r)
    equity.append(equity[-1] * (1 + r))

equity_curve = equity[-min(250, len(equity)):]

backtest_annual_return = (equity[-1] ** (252 / max(len(equity) - 1, 1)) - 1) if len(equity) > 1 else 0.0
backtest_sharpe = (np.mean(backtest_returns) / (np.std(backtest_returns) + 1e-9) * np.sqrt(252)) if backtest_returns else 0.0

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
        w: float(window_preds[w][pick])
        for w in WINDOWS
        if w in window_preds and pick in window_preds[w]
    },
    "samples": {k: v.tolist() for k, v in all_samples.items() if len(v) > 0},
    "equity_curve": equity_curve,
    "agreement": agreement,
    "backtest_metrics": {
        "annual_return": backtest_annual_return,
        "sharpe_ratio": backtest_sharpe,
        "total_days": len(backtest_returns),
        "final_equity": equity[-1] if equity else 1.0,
    },
}

os.makedirs("outputs", exist_ok=True)
fname = f"outputs/diffmap_{output['date']}.json"

with open(fname, "w") as f:
    json.dump(output, f, indent=2)

print("Saved locally:", fname)
print(f"Selected ETF: {pick} (confidence: {confidence:.2%})")
print(f"Backtest Sharpe: {backtest_sharpe:.2f}")

# ─────────────────────────────────────────────
# UPLOAD TO HUGGING FACE
# ─────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=os.path.basename(fname),
        repo_id=HF_OUTPUT_DATASET,
        repo_type="dataset",
    )
    print("Uploaded to HF:", HF_OUTPUT_DATASET)
else:
    print("Warning: HF_TOKEN not set, skipping upload")

# run.py

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from huggingface_hub import HfApi


from config import WINDOWS, ALL_ETFS, MACRO_VARS, LOOKBACK, HF_OUTPUT_DATASET
from portfolio import PortfolioState
from utils import compute_tbill_daily_rate
from calendar_utils import get_next_trading_day


# ─────────────────────────────────────────────
# TRAIN MODELS PER WINDOW + ETF
# ─────────────────────────────────────────────
models = {}

for w, start in WINDOWS.items():
    df_w = df[df["date"] >= start]
    models[w] = {}

    for etf in ALL_ETFS:
        models[w][etf] = train_model(df_w, etf)

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
        mu, _ = predict_etf(models[w][etf], df, etf)
        window_preds[w][etf] = mu

        data = df[[etf] + MACRO_VARS].dropna().values
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
        mu = window_preds[w][etf]
        mus.append(mu)

        s = window_samples[w][etf]
        combined.extend(s)

        if mu > 0:
            pos += 1

    final_scores[etf] = np.mean(mus)
    agreement[etf] = pos
    all_samples[etf] = np.array(combined)

# ─────────────────────────────────────────────
# TOP 3
# ─────────────────────────────────────────────
sorted_etfs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

top3 = [{"etf": e, "mu": float(m)} for e, m in sorted_etfs[:3]]

# ─────────────────────────────────────────────
# PORTFOLIO DECISION
# ─────────────────────────────────────────────
portfolio = PortfolioState()
tbill = compute_tbill_daily_rate(df)

pick, score = portfolio.decide(final_scores.copy(), all_samples, tbill)

mode = "NORMAL"
if portfolio.in_cash:
    mode = "CASH"
if portfolio.check_tsl():
    mode = "TSL_ACTIVE"

samples_best = all_samples.get(pick, np.array([0]))
confidence = float((samples_best > 0).mean())

# ─────────────────────────────────────────────
# LIGHT BACKTEST (FAST)
# ─────────────────────────────────────────────
equity = [1.0]
returns_df = df.set_index("date")[ALL_ETFS].pct_change().dropna()

portfolio_bt = PortfolioState()

for i in range(LOOKBACK, len(returns_df) - 1):
    row = returns_df.iloc[i]
    next_row = returns_df.iloc[i + 1]

    preds_bt = row.to_dict()

    samples_bt = {
        k: np.random.normal(v, 0.01, 50)
        for k, v in preds_bt.items()
    }

    tbill_bt = compute_tbill_daily_rate(df.iloc[:i])

    pick_bt, _ = portfolio_bt.decide(preds_bt, samples_bt, tbill_bt)

    if pick_bt == "CASH":
        r = tbill_bt
    else:
        r = next_row[pick_bt]

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
    "samples": {k: v.tolist() for k, v in all_samples.items()},
    "equity_curve": equity_curve,
    "agreement": agreement
}

os.makedirs("outputs", exist_ok=True)
fname = f"outputs/diffmap_{output['date']}.json"

with open(fname, "w") as f:
    json.dump(output, f, indent=2)

print("Saved locally:", fname)

# ─────────────────────────────────────────────
# UPLOAD TO HUGGING FACE
# ─────────────────────────────────────────────
api = HfApi(token=os.environ.get("HF_TOKEN"))

api.upload_file(
    path_or_fileobj=fname,
    path_in_repo=os.path.basename(fname),
    repo_id=HF_OUTPUT_DATASET,
    repo_type="dataset"
)

print("Uploaded to HF:", HF_OUTPUT_DATASET)

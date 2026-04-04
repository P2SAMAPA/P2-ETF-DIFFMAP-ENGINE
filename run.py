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
from config import WINDOWS, ALL_ETFS, MACRO_VARS, LOOKBACK, HF_OUTPUT_DATASET, FI_ETFS, EQ_ETFS
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

# Store dates for later use in output
data_dates = df.index.strftime("%Y-%m-%d").tolist()

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
# AGGREGATION — overall + per group
# ─────────────────────────────────────────────
final_scores = {}
agreement = {}
all_samples = {}
window_scores_by_etf = {w: {} for w in WINDOWS}  # per-etf per-window scores

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
        window_scores_by_etf[w][etf] = mu

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

# Best EQ and FI picks
eq_scores = {k: v for k, v in final_scores.items() if k in EQ_ETFS}
fi_scores = {k: v for k, v in final_scores.items() if k in FI_ETFS}

eq_pick = max(eq_scores, key=eq_scores.get) if eq_scores else None
fi_pick = max(fi_scores, key=fi_scores.get) if fi_scores else None

# ─────────────────────────────────────────────
# OVERALL PORTFOLIO DECISION
# ─────────────────────────────────────────────
sorted_etfs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
top3 = [{"etf": e, "mu": float(m)} for e, m in sorted_etfs[:3]]

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
# BACKTEST — separate EQ and FI curves
# ─────────────────────────────────────────────
returns_df = df[ALL_ETFS].dropna()

# Helper to run backtest for a subset of ETFs
def run_backtest(etf_subset, returns_df, tbill_fn, df_full):
    equity = [1.0]
    port = PortfolioState()
    bt_returns = []

    subset_returns = returns_df[etf_subset].dropna()

    for i in range(LOOKBACK, len(subset_returns) - 1):
        row = subset_returns.iloc[i]
        next_row = subset_returns.iloc[i + 1]

        preds_bt = row.to_dict()

        samples_bt = {}
        for k, v in preds_bt.items():
            hist_vol = subset_returns[k].rolling(20).std().iloc[i] if i >= 20 else 0.01
            samples_bt[k] = np.random.normal(v, abs(hist_vol) + 0.001, 50)

        tbill_bt = tbill_fn(df_full.reset_index().iloc[:i + LOOKBACK])
        pick_bt, _ = port.decide(preds_bt, samples_bt, tbill_bt)

        if pick_bt == "CASH":
            r = tbill_bt
        else:
            r = next_row[pick_bt] if pick_bt in next_row.index else 0.0

        port.update_returns(r)
        bt_returns.append(r)
        equity.append(equity[-1] * (1 + r))

    return equity, bt_returns

print("Running EQ backtest...")
eq_equity, eq_returns = run_backtest(EQ_ETFS, returns_df, compute_tbill_daily_rate, df)

print("Running FI backtest...")
fi_equity, fi_returns = run_backtest(FI_ETFS, returns_df, compute_tbill_daily_rate, df)

# Benchmark curves — SPY and AGG (simple buy-and-hold)
spy_col = "SPY_ret" if "SPY_ret" in returns_df.columns else None
agg_col = "AGG_ret" if "AGG_ret" in df.columns else None

def buy_hold_equity(col, df_src):
    if col is None or col not in df_src.columns:
        return [1.0]
    rets = df_src[col].dropna().values[LOOKBACK:]
    eq = [1.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    return eq

spy_equity = buy_hold_equity(spy_col, returns_df)
agg_equity = buy_hold_equity(agg_col, df)

# Align all curves to same length (min)
min_len = min(len(eq_equity), len(fi_equity), len(spy_equity), len(agg_equity))
eq_equity  = eq_equity[:min_len]
fi_equity  = fi_equity[:min_len]
spy_equity = spy_equity[:min_len]
agg_equity = agg_equity[:min_len]

# Dates for equity curve x-axis
curve_dates = data_dates[LOOKBACK:LOOKBACK + min_len] if len(data_dates) >= LOOKBACK + min_len else data_dates[:min_len]

# Backtest metrics
def bt_metrics(equity_list, returns_list):
    if len(equity_list) < 2 or not returns_list:
        return {"annual_return": 0, "sharpe_ratio": 0, "total_days": 0, "final_equity": 1.0}
    annual = equity_list[-1] ** (252 / max(len(equity_list) - 1, 1)) - 1
    sharpe = np.mean(returns_list) / (np.std(returns_list) + 1e-9) * np.sqrt(252)
    return {
        "annual_return": float(annual),
        "sharpe_ratio": float(sharpe),
        "total_days": len(returns_list),
        "final_equity": float(equity_list[-1]),
    }

# ─────────────────────────────────────────────
# WINDOW SCORES — per ETF group per window
# ─────────────────────────────────────────────
window_table = {}
for w, start in WINDOWS.items():
    year = start[:4]
    eq_best_score = max((window_scores_by_etf[w].get(e, float("-inf")) for e in EQ_ETFS), default=0.0)
    fi_best_score = max((window_scores_by_etf[w].get(e, float("-inf")) for e in FI_ETFS), default=0.0)
    eq_best_etf   = max(EQ_ETFS, key=lambda e: window_scores_by_etf[w].get(e, float("-inf")))
    fi_best_etf   = max(FI_ETFS, key=lambda e: window_scores_by_etf[w].get(e, float("-inf")))
    window_table[w] = {
        "start_year": year,
        "eq_pick":  eq_best_etf,
        "eq_score": float(eq_best_score),
        "fi_pick":  fi_best_etf,
        "fi_score": float(fi_best_score),
    }

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
    "eq_pick": eq_pick,
    "fi_pick": fi_pick,
    "eq_score": float(eq_scores.get(eq_pick, 0)) if eq_pick else 0.0,
    "fi_score": float(fi_scores.get(fi_pick, 0)) if fi_pick else 0.0,
    "eq_confidence": float((all_samples.get(eq_pick, np.array([0])) > 0).mean()) if eq_pick else 0.0,
    "fi_confidence": float((all_samples.get(fi_pick, np.array([0])) > 0).mean()) if fi_pick else 0.0,
    "samples": {k: v.tolist() for k, v in all_samples.items() if len(v) > 0},
    "curve_dates": curve_dates,
    "equity_curves": {
        "eq":  eq_equity,
        "fi":  fi_equity,
        "spy": spy_equity,
        "agg": agg_equity,
    },
    "agreement": agreement,
    "window_table": window_table,
    "window_scores": {
        w: float(window_preds[w][pick]) for w in WINDOWS
        if w in window_preds and pick in window_preds[w]
    },
    "backtest_eq":  bt_metrics(eq_equity,  eq_returns),
    "backtest_fi":  bt_metrics(fi_equity,  fi_returns),
    "signal_history_eq": eq_pick,
    "signal_history_fi": fi_pick,
}

os.makedirs("outputs", exist_ok=True)
fname = f"outputs/diffmap_{output['date']}.json"

with open(fname, "w") as f:
    json.dump(output, f, indent=2)

print("Saved locally:", fname)
print(f"Overall pick: {pick} | EQ: {eq_pick} | FI: {fi_pick}")

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

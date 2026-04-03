import os
import json
import pandas as pd
from datetime import datetime

from data_loader import load_data
from train import train_model
from infer import predict_all
from utils import compute_tbill_daily_rate
from portfolio import PortfolioState
from calendar_utils import get_next_trading_day
from config import WINDOWS, ALL_ETFS

# Load data
df = load_data()

# Train models per window
models = {}
for k, start in WINDOWS.items():
    df_w = df[df["date"] >= start]
    models[k] = train_model(df_w, ALL_ETFS[0])  # shared model

# Predict
preds, samples = predict_all(models, df)

# Portfolio logic
portfolio = PortfolioState()
tbill = compute_tbill_daily_rate(df)

pick, score = portfolio.decide(preds, samples, tbill)

# Next trading day
next_day = get_next_trading_day()

# Output
output = {
    "engine": "DIFFMAP-ETF",
    "date": datetime.utcnow().strftime("%Y-%m-%d"),
    "next_trading_day": next_day,
    "pick": pick,
    "score": float(score),
}

os.makedirs("outputs", exist_ok=True)

fname = f"outputs/diffmap_{output['date']}.json"

with open(fname, "w") as f:
    json.dump(output, f, indent=2)

print(output)

import numpy as np

def aggregate_scores(window_preds):
    mus = np.array([v[0] for v in window_preds.values()])
    p_up = np.array([v[1] for v in window_preds.values()])
    mu_mean = mus.mean()
    confidence = p_up.mean()
    return mu_mean * confidence

def compute_tbill_daily_rate(df):
    # TBILL_daily in parquet is already a daily rate
    if "TBILL_daily" in df.columns:
        val = df["TBILL_daily"].iloc[-1]
        if not np.isnan(val):
            return float(val)
    return 0.04 / 252  # fallback ~4% annualized

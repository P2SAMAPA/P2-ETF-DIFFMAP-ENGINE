import numpy as np

def aggregate_scores(window_preds):
    mus = np.array([v[0] for v in window_preds.values()])
    p_up = np.array([v[1] for v in window_preds.values()])
    mu_mean = mus.mean()
    confidence = p_up.mean()
    return mu_mean * confidence

def compute_tbill_daily_rate(df):
    # TBILL_daily is already a daily rate (not annualized %)
    # Column renamed from DTB3 to TBILL_daily in macro_derived.parquet
    if "TBILL_daily" in df.columns:
        return float(df["TBILL_daily"].iloc[-1])
    # Fallback: ~4% annualized
    return 0.04 / 252

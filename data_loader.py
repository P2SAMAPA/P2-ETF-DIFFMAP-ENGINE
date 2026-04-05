import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from config import ALL_ETFS, MACRO_VARS

HF_DATASET = "P2SAMAPA/p2-etf-deepm-data"
MIN_ROWS = 100

def _ensure_date_column(df, label):
    """Robustly extracts date into a plain 'date' column."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    for col in ["Date", "DATE", "datetime", "Datetime", "time"]:
        if col in df.columns:
            df = df.rename(columns={col: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df

    if isinstance(df.index, pd.DatetimeIndex) or df.index.name in ("date", "Date", "DATE"):
        df = df.reset_index().rename(columns={df.index.name or df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    
    raise ValueError(f"[{label}] Could not find a date column or index.")

def load_data():
    # ── DOWNLOAD ─────────────────────────────────
    try:
        returns_path = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename="data/etf_returns.parquet")
        macro_path = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename="data/macro_derived.parquet")
    except Exception as e:
        print(f"[data_loader] HF Download failed: {e}")
        raise

    df_ret = pd.read_parquet(returns_path)
    df_macro = pd.read_parquet(macro_path)

    # ── DATE ALIGNMENT ───────────────────────────
    df_ret = _ensure_date_column(df_ret, "returns")
    df_macro = _ensure_date_column(df_macro, "macro")

    df_ret["date"] = df_ret["date"].dt.normalize()
    df_macro["date"] = df_macro["date"].dt.normalize()

    # ── MERGE & RECOVERY ─────────────────────────
    # Use outer join first to handle frequency mismatches, then forward fill macro
    df = pd.merge(df_ret, df_macro, on="date", how="left")
    
    # Sort by date before filling to ensure temporal integrity
    df = df.sort_values("date").reset_index(drop=True)
    
    # Forward fill macro variables (last known value) but don't fill ETF returns
    df[MACRO_VARS] = df[MACRO_VARS].ffill()
    
    # Final drop for rows that are still NaN (e.g., start of dataset)
    df = df.dropna(subset=ALL_ETFS + MACRO_VARS)

    # ── FINANCIAL HYGIENE: WINSORIZATION ─────────
    # Clip extreme outliers (3 standard deviations) to prevent diffusion 'explosions'
    for col in ALL_ETFS + MACRO_VARS:
        upper = df[col].quantile(0.995)
        lower = df[col].quantile(0.005)
        df[col] = df[col].clip(lower, upper)

    # ── FEATURE SCALING ──────────────────────────
    # Standardize features: (x - mean) / std
    # For a live model, you'd ideally use a rolling mean, 
    # but for this batch loader, we standardize globally.
    for col in ALL_ETFS + MACRO_VARS:
        mu, sigma = df[col].mean(), df[col].std()
        if sigma != 0:
            df[col] = (df[col] - mu) / sigma

    # ── GUARD ────────────────────────────────────
    if len(df) < MIN_ROWS:
        raise ValueError(f"Dataset too small after cleaning: {len(df)} rows.")

    needed_cols = ["date"] + ALL_ETFS + MACRO_VARS
    return df[needed_cols].reset_index(drop=True)

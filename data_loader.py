# data_loader.py

import pandas as pd
from huggingface_hub import hf_hub_download

from config import ALL_ETFS, MACRO_VARS

HF_DATASET = "P2SAMAPA/p2-etf-deepm-data"


def _ensure_date_column(df, name):
    """
    Ensures a 'date' column exists in dataframe.
    Handles:
    - date as index
    - different column names (Date, datetime, etc.)
    """

    # Case 1: already exists
    if "date" in df.columns:
        return df

    # Case 2: common alternatives
    for col in ["Date", "DATE", "datetime", "Datetime", "time"]:
        if col in df.columns:
            df = df.rename(columns={col: "date"})
            return df

    # Case 3: date is index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if df.columns[0] != "date":
            df = df.rename(columns={df.columns[0]: "date"})
        return df

    # Case 4: fallback
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    return df


def load_data():
    """
    Loads ETF returns + macro features from HF parquet dataset.
    Returns a clean dataframe ready for model training.
    """

    # ─────────────────────────────────────────────
    # LOAD ETF RETURNS
    # ─────────────────────────────────────────────
    returns_path = hf_hub_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        filename="data/etf_returns.parquet"
    )

    df_ret = pd.read_parquet(returns_path)

    # ─────────────────────────────────────────────
    # LOAD MACRO FEATURES
    # ─────────────────────────────────────────────
    macro_path = hf_hub_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        filename="data/macro_derived.parquet"
    )

    df_macro = pd.read_parquet(macro_path)

    # ─────────────────────────────────────────────
    # FIX DATE COLUMN (ROBUST)
    # ─────────────────────────────────────────────
    df_ret = _ensure_date_column(df_ret, "returns")
    df_macro = _ensure_date_column(df_macro, "macro")

    # ensure datetime
    df_ret["date"] = pd.to_datetime(df_ret["date"], errors="coerce")
    df_macro["date"] = pd.to_datetime(df_macro["date"], errors="coerce")

    # drop bad dates early
    df_ret = df_ret.dropna(subset=["date"])
    df_macro = df_macro.dropna(subset=["date"])

    # ─────────────────────────────────────────────
    # MERGE
    # ─────────────────────────────────────────────
    df = pd.merge(df_ret, df_macro, on="date", how="inner")

    # ─────────────────────────────────────────────
    # COLUMN FILTERING
    # ─────────────────────────────────────────────
    needed_cols = ["date"] + ALL_ETFS + MACRO_VARS

    missing = [c for c in needed_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[needed_cols]

    # ─────────────────────────────────────────────
    # CLEANING
    # ─────────────────────────────────────────────
    df = df.replace([float("inf"), -float("inf")], pd.NA)
    df = df.dropna()

    # sort + reset index
    df = df.sort_values("date").reset_index(drop=True)

    return df

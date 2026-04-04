# data_loader.py

import pandas as pd
from huggingface_hub import hf_hub_download

from config import ALL_ETFS, MACRO_VARS

HF_DATASET = "P2SAMAPA/p2-etf-deepm-data"


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
    # BASIC VALIDATION
    # ─────────────────────────────────────────────
    if "date" not in df_ret.columns:
        raise ValueError("etf_returns.parquet missing 'date' column")

    if "date" not in df_macro.columns:
        raise ValueError("macro_derived.parquet missing 'date' column")

    # ─────────────────────────────────────────────
    # MERGE
    # ─────────────────────────────────────────────
    df = pd.merge(df_ret, df_macro, on="date", how="inner")

    # ensure datetime
    df["date"] = pd.to_datetime(df["date"])

    # ─────────────────────────────────────────────
    # COLUMN FILTERING (CRITICAL)
    # ─────────────────────────────────────────────
    needed_cols = ["date"] + ALL_ETFS + MACRO_VARS

    missing = [c for c in needed_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[needed_cols]

    # ─────────────────────────────────────────────
    # CLEANING
    # ─────────────────────────────────────────────
    df = df.replace([float("inf"), -float("inf")], pd.NA)
    df = df.dropna()

    # sort
    df = df.sort_values("date").reset_index(drop=True)

    return df

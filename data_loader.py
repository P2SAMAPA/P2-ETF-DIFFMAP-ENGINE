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
    # BASIC VALIDATION - CHECK FOR DATE COLUMN OR INDEX
    # ─────────────────────────────────────────────
    # CORRECTED: Handle case where date is the index, not a column
    if "date" not in df_ret.columns:
        # Check if the index is a datetime index and name it 'date'
        if pd.api.types.is_datetime64_any_dtype(df_ret.index):
            df_ret = df_ret.reset_index()
            # Rename the index column to 'date' if it's not already named
            if df_ret.columns[0] not in ['date', 'Date', 'DATE']:
                df_ret = df_ret.rename(columns={df_ret.columns[0]: 'date'})
        else:
            raise ValueError("etf_returns.parquet missing 'date' column and index is not datetime")

    if "date" not in df_macro.columns:
        # Check if the index is a datetime index and name it 'date'
        if pd.api.types.is_datetime64_any_dtype(df_macro.index):
            df_macro = df_macro.reset_index()
            # Rename the index column to 'date' if it's not already named
            if df_macro.columns[0] not in ['date', 'Date', 'DATE']:
                df_macro = df_macro.rename(columns={df_macro.columns[0]: 'date'})
        else:
            raise ValueError("macro_derived.parquet missing 'date' column and index is not datetime")

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

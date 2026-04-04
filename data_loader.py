import pandas as pd
from huggingface_hub import hf_hub_download
from config import ALL_ETFS, MACRO_VARS

HF_DATASET = "P2SAMAPA/p2-etf-deepm-data"
MIN_ROWS = 100  # need enough history to train


def _ensure_date_column(df, name):
    if "date" in df.columns:
        return df
    for col in ["Date", "DATE", "datetime", "Datetime", "time"]:
        if col in df.columns:
            df = df.rename(columns={col: "date"})
            return df
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if df.columns[0] != "date":
            df = df.rename(columns={df.columns[0]: "date"})
        return df
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    return df


def load_data():
    # ── ETF RETURNS ──────────────────────────────
    returns_path = hf_hub_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        filename="data/etf_returns.parquet"
    )
    df_ret = pd.read_parquet(returns_path)
    print(f"[data_loader] etf_returns rows: {len(df_ret)}, cols: {list(df_ret.columns)}")

    # ── MACRO FEATURES ───────────────────────────
    macro_path = hf_hub_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        filename="data/macro_derived.parquet"
    )
    df_macro = pd.read_parquet(macro_path)
    print(f"[data_loader] macro_derived rows: {len(df_macro)}, cols: {list(df_macro.columns)}")

    # ── FIX DATE COLUMNS ─────────────────────────
    df_ret = _ensure_date_column(df_ret, "returns")
    df_macro = _ensure_date_column(df_macro, "macro")

    df_ret["date"] = pd.to_datetime(df_ret["date"], errors="coerce")
    df_macro["date"] = pd.to_datetime(df_macro["date"], errors="coerce")

    df_ret = df_ret.dropna(subset=["date"])
    df_macro = df_macro.dropna(subset=["date"])

    # ── MERGE ────────────────────────────────────
    df = pd.merge(df_ret, df_macro, on="date", how="inner")
    print(f"[data_loader] merged rows: {len(df)}")

    # ── GUARD: not enough rows ───────────────────
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Dataset too small after merge: {len(df)} rows. "
            f"Need at least {MIN_ROWS}. "
            f"Your HuggingFace parquet at {HF_DATASET} is not being "
            f"appended correctly — it is overwriting with only today's row. "
            f"Fix the data upload script to append historical rows."
        )

    # ── COLUMN FILTERING ─────────────────────────
    needed_cols = ["date"] + ALL_ETFS + MACRO_VARS
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[needed_cols]

    # ── CLEANING ─────────────────────────────────
    df = df.replace([float("inf"), -float("inf")], pd.NA)
    df = df.dropna()
    df = df.sort_values("date").reset_index(drop=True)

    return df

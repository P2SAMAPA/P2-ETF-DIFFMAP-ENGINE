import pandas as pd
from huggingface_hub import hf_hub_download
from config import ALL_ETFS, MACRO_VARS

HF_DATASET = "P2SAMAPA/p2-etf-deepm-data"
MIN_ROWS = 100


def _ensure_date_column(df, label):
    """
    Robustly extracts date into a plain 'date' column.
    Handles: date as index (named or unnamed), or as a column.
    """
    # Case 1: already a column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    # Case 2: common alternative column names
    for col in ["Date", "DATE", "datetime", "Datetime", "time"]:
        if col in df.columns:
            df = df.rename(columns={col: "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df

    # Case 3: index is a DatetimeIndex or named index
    if isinstance(df.index, pd.DatetimeIndex) or df.index.name in ("date", "Date", "DATE", "datetime", "time"):
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        print(f"  [{label}] extracted date from index, sample: {df['date'].iloc[:3].tolist()}")
        return df

    # Case 4: unnamed index that looks like dates
    try:
        test = pd.to_datetime(df.index)
        df.index = test
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        print(f"  [{label}] coerced unnamed index to date, sample: {df['date'].iloc[:3].tolist()}")
        return df
    except Exception:
        pass

    raise ValueError(
        f"[{label}] Could not find a date column or index. "
        f"Columns: {list(df.columns)}, Index: {df.index[:3].tolist()}"
    )


def load_data():
    # ── ETF RETURNS ──────────────────────────────
    returns_path = hf_hub_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        filename="data/etf_returns.parquet"
    )
    df_ret = pd.read_parquet(returns_path)
    print(f"[data_loader] etf_returns rows: {len(df_ret)}, cols: {list(df_ret.columns)}")
    print(f"[data_loader] etf_returns index: type={type(df_ret.index)}, name={df_ret.index.name}, sample={df_ret.index[:3].tolist()}")

    # ── MACRO FEATURES ───────────────────────────
    macro_path = hf_hub_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        filename="data/macro_derived.parquet"
    )
    df_macro = pd.read_parquet(macro_path)
    print(f"[data_loader] macro_derived rows: {len(df_macro)}, cols: {list(df_macro.columns)}")
    print(f"[data_loader] macro_derived index: type={type(df_macro.index)}, name={df_macro.index.name}, sample={df_macro.index[:3].tolist()}")

    # ── FIX DATE COLUMNS ─────────────────────────
    df_ret = _ensure_date_column(df_ret, "returns")
    df_macro = _ensure_date_column(df_macro, "macro")

    # Normalize to date-only (strip time component) so merge aligns
    df_ret["date"] = pd.to_datetime(df_ret["date"]).dt.normalize()
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.normalize()

    df_ret = df_ret.dropna(subset=["date"])
    df_macro = df_macro.dropna(subset=["date"])

    print(f"[data_loader] returns date range: {df_ret['date'].min()} → {df_ret['date'].max()}")
    print(f"[data_loader] macro date range:   {df_macro['date'].min()} → {df_macro['date'].max()}")

    # ── MERGE ────────────────────────────────────
    df = pd.merge(df_ret, df_macro, on="date", how="inner")
    print(f"[data_loader] merged rows: {len(df)}")

    # ── GUARD: not enough rows ───────────────────
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Dataset too small after merge: {len(df)} rows, need at least {MIN_ROWS}.\n"
            f"Returns date sample: {df_ret['date'].iloc[:3].tolist()}\n"
            f"Macro date sample:   {df_macro['date'].iloc[:3].tolist()}\n"
            f"This is a date alignment issue — check timezone or format differences."
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

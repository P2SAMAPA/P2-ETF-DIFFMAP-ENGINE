import pandas as pd
from huggingface_hub import hf_hub_download

def _ensure_date_column(df, label):
    """
    Ensures the DataFrame has a proper 'date' column (datetime).
    Handles:
    - Columns named: 'date', 'Date', 'DATE', 'datetime', 'Datetime', 'time', 'TIME', 'timestamp', 'TIMESTAMP'
    - A column named 'index' that contains milliseconds
    - The DataFrame index if it is a DatetimeIndex or numeric milliseconds
    """
    date_candidates = ["date", "Date", "DATE", "datetime", "Datetime", "time", "TIME", "timestamp", "TIMESTAMP"]
    for col in date_candidates:
        if col in df.columns:
            df = df.rename(columns={col: "date"})
            if pd.api.types.is_numeric_dtype(df["date"]):
                unit = 'ms' if df["date"].iloc[0] > 1e12 else 's'
                df["date"] = pd.to_datetime(df["date"], unit=unit, errors="coerce")
            else:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df

    if "index" in df.columns:
        df = df.rename(columns={"index": "date"})
        if pd.api.types.is_numeric_dtype(df["date"]):
            unit = 'ms' if df["date"].iloc[0] > 1e12 else 's'
            df["date"] = pd.to_datetime(df["date"], unit=unit, errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "date"})
        return df

    if pd.api.types.is_numeric_dtype(df.index):
        first_val = df.index[0]
        if first_val > 1e12:
            df.index = pd.to_datetime(df.index, unit='ms')
        elif first_val > 1e9:
            df.index = pd.to_datetime(df.index, unit='s')
        else:
            raise ValueError(f"[{label}] Numeric index but value {first_val} is not a valid timestamp (ms or s).")
        df = df.reset_index().rename(columns={"index": "date"})
        return df

    raise ValueError(f"[{label}] Could not find a date column or index. Available columns: {df.columns.tolist()}")

def load_data():
    """
    Load ETF and macro data from Hugging Face dataset, merge them on 'date',
    and return a single DataFrame ready for training.
    """
    # Download ETF data
    etf_path = hf_hub_download(
        repo_id="P2SAMAPA/p2-etf-deepm-data",
        repo_type="dataset",
        filename="data/etf_returns.parquet"
    )
    df_etf = pd.read_parquet(etf_path)
    df_etf = _ensure_date_column(df_etf, "etf")

    # Download macro data
    macro_path = hf_hub_download(
        repo_id="P2SAMAPA/p2-etf-deepm-data",
        repo_type="dataset",
        filename="data/macro_derived.parquet"
    )
    df_macro = pd.read_parquet(macro_path)
    df_macro = _ensure_date_column(df_macro, "macro")

    # Drop rows with missing dates
    df_etf = df_etf.dropna(subset=["date"])
    df_macro = df_macro.dropna(subset=["date"])

    # Ensure timezone-naive for merging
    df_etf["date"] = pd.to_datetime(df_etf["date"]).dt.tz_localize(None)
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None)

    # Merge on date (inner join keeps only dates present in both)
    df_merged = pd.merge(df_etf, df_macro, on="date", how="inner")

    # Optional: sort by date
    df_merged = df_merged.sort_values("date").reset_index(drop=True)

    return df_merged

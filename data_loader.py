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
    # 1. Look for a known date column name
    date_candidates = ["date", "Date", "DATE", "datetime", "Datetime", "time", "TIME", "timestamp", "TIMESTAMP"]
    for col in date_candidates:
        if col in df.columns:
            df = df.rename(columns={col: "date"})
            # Convert to datetime, handling numeric milliseconds/seconds
            if pd.api.types.is_numeric_dtype(df["date"]):
                # Assume milliseconds if values > 1e12, otherwise seconds
                unit = 'ms' if df["date"].iloc[0] > 1e12 else 's'
                df["date"] = pd.to_datetime(df["date"], unit=unit, errors="coerce")
            else:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df

    # 2. Check for a column named 'index' (common when reset_index was used)
    if "index" in df.columns:
        df = df.rename(columns={"index": "date"})
        if pd.api.types.is_numeric_dtype(df["date"]):
            unit = 'ms' if df["date"].iloc[0] > 1e12 else 's'
            df["date"] = pd.to_datetime(df["date"], unit=unit, errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    # 3. Check if the index is a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "date"})
        return df

    # 4. If index is numeric and looks like milliseconds/seconds, convert it
    if pd.api.types.is_numeric_dtype(df.index):
        first_val = df.index[0]
        if first_val > 1e12:   # milliseconds
            df.index = pd.to_datetime(df.index, unit='ms')
        elif first_val > 1e9:  # seconds
            df.index = pd.to_datetime(df.index, unit='s')
        else:
            raise ValueError(f"[{label}] Numeric index but value {first_val} is not a valid timestamp (ms or s).")
        df = df.reset_index().rename(columns={"index": "date"})
        return df

    # 5. If no date column or convertible index found, raise error
    raise ValueError(f"[{label}] Could not find a date column or index. Available columns: {df.columns.tolist()}")

def load_data():
    """
    Load ETF and macro data from Hugging Face dataset.
    Returns:
        df_etf: DataFrame with ETF returns and date column
        df_macro: DataFrame with macro features and date column
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

    # Optional: Drop rows with NaN dates (if any)
    df_etf = df_etf.dropna(subset=["date"])
    df_macro = df_macro.dropna(subset=["date"])

    # Ensure dates are timezone-naive for merging
    df_etf["date"] = pd.to_datetime(df_etf["date"]).dt.tz_localize(None)
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None)

    return df_etf, df_macro

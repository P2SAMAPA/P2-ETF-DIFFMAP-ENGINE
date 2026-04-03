import pandas as pd
from datasets import load_dataset
from config import HF_DATASET

def load_data():
    ds = load_dataset(HF_DATASET)

    returns = pd.DataFrame(ds["train"]["etf_returns"])
    macro = pd.DataFrame(ds["train"]["macro_derived"])

    df = returns.merge(macro, on="date", how="inner")
    df = df.sort_values("date")

    return df

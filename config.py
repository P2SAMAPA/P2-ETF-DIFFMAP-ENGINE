# config.py

HF_DATASET = "P2SAMAPA/p2-etf-deepm-data"
HF_OUTPUT_DATASET = "P2SAMAPA/p2-etf-diffmap-results"

# ETFs
FI_ETFS = ["TLT","LQD","HYG","VNQ","GLD","SLV","PFF","MBB"]
FI_BENCHMARK = "AGG"

EQ_ETFS = ["SPY","QQQ","XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","GDX","XME"]
EQ_BENCHMARK = "SPY"

ALL_ETFS = FI_ETFS + EQ_ETFS

# Macro
MACRO_VARS = ["VIX","T10Y2Y","HY_SPREAD","USD_INDEX","DTB3"]

# Windows
WINDOWS = {
    "A": "2008-01-01",
    "B": "2012-01-01",
    "C": "2015-01-01",
    "D": "2017-01-01",
    "E": "2019-01-01",
    "F": "2021-01-01",
    "G": "2023-01-01",
}

# Model
LOOKBACK = 30
N_SAMPLES = 100
SIGMA_MIN = 0.01
SIGMA_MAX = 0.2

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3

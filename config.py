HF_DATASET = "P2SAMAPA/p2-etf-deepm-data"
HF_OUTPUT_DATASET = "P2SAMAPA/p2-etf-diffmap-results"

FI_ETFS = ["TLT_ret","LQD_ret","HYG_ret","VNQ_ret","GLD_ret","SLV_ret","PFF_ret","MBB_ret"]
FI_BENCHMARK = "AGG_ret"
EQ_ETFS = ["QQQ_ret","XLK_ret","XLF_ret","XLE_ret","XLV_ret","XLI_ret","XLY_ret","XLP_ret","XLU_ret","GDX_ret","XME_ret"]
EQ_BENCHMARK = "SPY_ret"
ALL_ETFS = FI_ETFS + EQ_ETFS

MACRO_VARS = [
    "VIX_zscore",
    "VIX_chg1d",
    "YC_slope",
    "HY_spread_zscore",
    "USD_zscore",
    "TBILL_daily",
    "credit_stress",
    "macro_stress_composite",
]

WINDOWS = {
    "A": "2008-01-01",
    "B": "2009-01-01",
    "C": "2010-01-01",
    "D": "2011-01-01",
    "E": "2012-01-01",
    "F": "2012-01-01",
    "G": "2013-01-01",
    "H": "2014-01-01",
    "I": "2015-01-01",
    "J": "2016-01-01",
    "K": "2017-01-01",
    "L": "2018-01-01",
    "M": "2019-01-01",
    "N": "2020-01-01",
    "O": "2021-01-01",
    "P": "2022-01-01",
    "Q": "2023-01-01",
    "R": "2024-01-01",
    "S": "2025-01-01",
}

LOOKBACK = 30
N_SAMPLES = 100
SIGMA_MIN = 0.01
SIGMA_MAX = 0.2
EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3

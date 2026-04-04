# app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download

st.set_page_config(layout="wide")

HF_REPO = "P2SAMAPA/p2-etf-diffmap-results"


def clean_etf_name(name: str) -> str:
    """Strip _ret or _logret suffix for display."""
    return name.replace("_ret", "").replace("_logret", "") if isinstance(name, str) else name


# ─────────────────────────────────────────────
# LOAD LATEST SIGNAL FROM HF
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_latest():
    api = HfApi()
    files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    json_files = sorted([f for f in files if f.endswith(".json")])
    if not json_files:
        return None
    latest = json_files[-1]
    path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=latest
    )
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# LOAD HISTORY FROM HF
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_history():
    api = HfApi()
    files = sorted([
        f for f in api.list_repo_files(HF_REPO, repo_type="dataset")
        if f.endswith(".json")
    ])
    history = []
    for f in files[-30:]:
        path = hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=f
        )
        with open(path) as file:
            d = json.load(file)
            history.append({
                "Date": d.get("date"),
                "Pick": clean_etf_name(d.get("pick")),
                "Mode": d.get("mode"),
                "Score": round(d.get("score", 0), 4)
            })
    return pd.DataFrame(history[::-1])


# ─────────────────────────────────────────────
# MAIN LOAD
# ─────────────────────────────────────────────
data = load_latest()

if data is None:
    st.warning("No signals yet. Run GitHub Action first.")
    st.stop()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("DIFFMAP — Diffusion ETF Engine")
st.caption("Generative Modeling · Multi-Window · Distribution-Based Selection")


# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────
pick = clean_etf_name(data.get("pick", "N/A"))
confidence = data.get("confidence", 0)
mode = data.get("mode", "N/A")
next_day = data.get("next_trading_day", "N/A")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(f"""
    <div style="padding:25px;border-radius:15px;background:#f4f0ff;">
        <h1>{pick}</h1>
        <h3 style="color:#6c4cff;">{round(confidence*100,1)}% conviction</h3>
        <p>Signal for <b>{next_day}</b></p>
        <p>Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Mode", mode)
    st.metric("Score", round(data.get("score", 0), 4))


# ─────────────────────────────────────────────
# TOP 3
# ─────────────────────────────────────────────
st.markdown("### Top Alternatives")

top3 = data.get("top_3", [])
cols = st.columns(3)

for i, t in enumerate(top3):
    cols[i].metric(
        label=clean_etf_name(t["etf"]),
        value=f"{round(t['mu']*100,2)}%"
    )


# ─────────────────────────────────────────────
# ALL ETF SCORES
# ─────────────────────────────────────────────
st.markdown("### All ETF Scores")

agreement = data.get("agreement", {})
window_scores = data.get("window_scores", {})

if agreement:
    # Separate FI and EQ ETFs for display
    fi_etfs = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "PFF", "MBB"]
    eq_etfs = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME"]

    rows = []
    for raw_name, wins in agreement.items():
        clean = clean_etf_name(raw_name)
        category = "Fixed Income" if clean in fi_etfs else "Equity"
        rows.append({
            "ETF": clean,
            "Category": category,
            "Positive Windows": wins,
        })

    df_agree = pd.DataFrame(rows).sort_values("Positive Windows", ascending=False)

    tab1, tab2, tab3 = st.tabs(["All", "Equity", "Fixed Income"])

    with tab1:
        st.dataframe(df_agree, use_container_width=True)
    with tab2:
        st.dataframe(df_agree[df_agree["Category"] == "Equity"], use_container_width=True)
    with tab3:
        st.dataframe(df_agree[df_agree["Category"] == "Fixed Income"], use_container_width=True)


# ─────────────────────────────────────────────
# DISTRIBUTION HISTOGRAMS
# ─────────────────────────────────────────────
st.markdown("### Return Distributions (Top 3)")

samples = data.get("samples", {})
top3_names = [t["etf"] for t in top3]
cols = st.columns(3)

for i, etf_raw in enumerate(top3_names[:3]):
    if etf_raw in samples:
        df_plot = pd.DataFrame({"returns": samples[etf_raw]})
        cols[i].caption(clean_etf_name(etf_raw))
        cols[i].bar_chart(df_plot)


# ─────────────────────────────────────────────
# EQUITY CURVE
# ─────────────────────────────────────────────
st.markdown("### Strategy Equity Curve")

equity = data.get("equity_curve", [])

if equity:
    df_eq = pd.DataFrame({"equity": equity})
    st.line_chart(df_eq)
else:
    st.info("Equity curve not available yet.")


# ─────────────────────────────────────────────
# BACKTEST METRICS
# ─────────────────────────────────────────────
st.markdown("### Backtest Metrics")

bt = data.get("backtest_metrics", {})

if bt:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annual Return", f"{round(bt.get('annual_return', 0)*100, 2)}%")
    c2.metric("Sharpe Ratio", round(bt.get('sharpe_ratio', 0), 2))
    c3.metric("Total Days", bt.get('total_days', 0))
    c4.metric("Final Equity", round(bt.get('final_equity', 1), 3))


# ─────────────────────────────────────────────
# SIGNAL HISTORY
# ─────────────────────────────────────────────
st.markdown("### Signal History")

df_hist = load_history()

if not df_hist.empty:
    st.dataframe(df_hist, use_container_width=True)
else:
    st.info("No history available yet.")


# ─────────────────────────────────────────────
# REFRESH BUTTON
# ─────────────────────────────────────────────
if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("DIFFMAP Engine · Research Use Only · Not Financial Advice")

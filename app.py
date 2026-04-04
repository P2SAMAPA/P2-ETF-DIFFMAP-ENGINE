# app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download

st.set_page_config(layout="wide", page_title="DIFFMAP ETF Engine")

HF_REPO = "P2SAMAPA/p2-etf-diffmap-results"

FI_ETFS  = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "PFF", "MBB"]
EQ_ETFS  = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME"]

def clean(name):
    return name.replace("_ret", "").replace("_logret", "") if isinstance(name, str) else name


@st.cache_data(ttl=300)
def load_latest():
    api = HfApi()
    files = sorted([f for f in api.list_repo_files(repo_id=HF_REPO, repo_type="dataset") if f.endswith(".json")])
    if not files:
        return None
    path = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=files[-1])
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_history():
    api = HfApi()
    files = sorted([f for f in api.list_repo_files(HF_REPO, repo_type="dataset") if f.endswith(".json")])
    rows = []
    for f in files[-30:]:
        path = hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=f)
        with open(path) as fh:
            d = json.load(fh)
        rows.append({
            "Date":  d.get("date"),
            "Pick":  clean(d.get("pick")),
            "Mode":  d.get("mode"),
            "Score": round(d.get("score", 0), 4),
        })
    return pd.DataFrame(rows[::-1])


# ── LOAD ────────────────────────────────────────────────────
data = load_latest()
if data is None:
    st.warning("No signals yet. Run GitHub Action first.")
    st.stop()

agreement   = data.get("agreement", {})
samples     = data.get("samples", {})
next_day    = data.get("next_trading_day", "N/A")
mode        = data.get("mode", "N/A")
confidence  = data.get("confidence", 0)
bt          = data.get("backtest_metrics", {})


# ── HELPERS ──────────────────────────────────────────────────
def best_in_group(group_tickers):
    """Return (raw_key, score) for the best ETF in a group."""
    candidates = {k: v for k, v in agreement.items() if clean(k) in group_tickers}
    if not candidates:
        return None, 0
    best_key = max(candidates, key=candidates.get)
    # use mean sample as score proxy
    s = samples.get(best_key, [0])
    return best_key, float(np.mean(s))


fi_key,  fi_score  = best_in_group(FI_ETFS)
eq_key,  eq_score  = best_in_group(EQ_ETFS)
fi_conf  = float((np.array(samples.get(fi_key,  [0])) > 0).mean()) if fi_key  else 0
eq_conf  = float((np.array(samples.get(eq_key,  [0])) > 0).mean()) if eq_key  else 0
fi_wins  = agreement.get(fi_key, 0)
eq_wins  = agreement.get(eq_key, 0)


# ── HEADER ──────────────────────────────────────────────────
st.title("DIFFMAP — Diffusion ETF Engine")
st.caption("Generative Modeling · Multi-Window · Distribution-Based Selection")
st.markdown(f"**Signal for:** {next_day} &nbsp;|&nbsp; **Mode:** {mode} &nbsp;|&nbsp; Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
st.divider()


# ── DUAL HERO ────────────────────────────────────────────────
col_eq, col_fi = st.columns(2)

with col_eq:
    st.markdown(
        f"""
        <div style="padding:24px;border-radius:14px;background:#eef4ff;border:1px solid #c0d4f5;">
            <p style="margin:0;font-size:13px;color:#555;font-weight:600;letter-spacing:1px;">EQUITY PICK</p>
            <h1 style="margin:4px 0 2px;color:#1a3c8f;">{clean(eq_key) if eq_key else "—"}</h1>
            <h3 style="margin:0;color:#2d5be3;">{round(eq_conf*100,1)}% conviction</h3>
            <p style="margin:6px 0 0;color:#444;">Score: {round(eq_score*100,3)}% &nbsp;|&nbsp; {eq_wins}/7 windows agree</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_fi:
    st.markdown(
        f"""
        <div style="padding:24px;border-radius:14px;background:#f0fff4;border:1px solid #b2dfcc;">
            <p style="margin:0;font-size:13px;color:#555;font-weight:600;letter-spacing:1px;">FIXED INCOME PICK</p>
            <h1 style="margin:4px 0 2px;color:#1a5c3a;">{clean(fi_key) if fi_key else "—"}</h1>
            <h3 style="margin:0;color:#2db36a;">{round(fi_conf*100,1)}% conviction</h3>
            <p style="margin:6px 0 0;color:#444;">Score: {round(fi_score*100,3)}% &nbsp;|&nbsp; {fi_wins}/7 windows agree</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ── TOP 3 ────────────────────────────────────────────────────
st.markdown("### Top 3 Overall")
top3 = data.get("top_3", [])
cols = st.columns(3)
for i, t in enumerate(top3):
    cols[i].metric(label=clean(t["etf"]), value=f"{round(t['mu']*100, 3)}%")

st.divider()


# ── ALL ETF SCORES ───────────────────────────────────────────
st.markdown("### All ETF Scores")

rows = []
for raw, wins in agreement.items():
    c = clean(raw)
    cat = "Fixed Income" if c in FI_ETFS else "Equity"
    mu  = float(np.mean(samples.get(raw, [0])))
    conf = float((np.array(samples.get(raw, [0])) > 0).mean())
    rows.append({"ETF": c, "Category": cat, "Positive Windows": wins,
                 "Avg Sample Return (%)": round(mu*100, 3),
                 "Conviction (%)": round(conf*100, 1)})

df_all = pd.DataFrame(rows).sort_values("Positive Windows", ascending=False)

tab_all, tab_eq, tab_fi = st.tabs(["All", "Equity", "Fixed Income"])
with tab_all:
    st.dataframe(df_all, use_container_width=True, hide_index=True)
with tab_eq:
    st.dataframe(df_all[df_all["Category"] == "Equity"], use_container_width=True, hide_index=True)
with tab_fi:
    st.dataframe(df_all[df_all["Category"] == "Fixed Income"], use_container_width=True, hide_index=True)

st.divider()


# ── DISTRIBUTIONS ────────────────────────────────────────────
st.markdown("### Return Distributions — Equity Pick vs FI Pick")
dcol1, dcol2 = st.columns(2)

if eq_key and eq_key in samples:
    dcol1.caption(f"{clean(eq_key)} (Equity)")
    dcol1.bar_chart(pd.DataFrame({"returns": samples[eq_key]}))

if fi_key and fi_key in samples:
    dcol2.caption(f"{clean(fi_key)} (Fixed Income)")
    dcol2.bar_chart(pd.DataFrame({"returns": samples[fi_key]}))

st.divider()


# ── EQUITY CURVE ─────────────────────────────────────────────
st.markdown("### Strategy Equity Curve")
equity = data.get("equity_curve", [])
if equity:
    st.line_chart(pd.DataFrame({"equity": equity}))
else:
    st.info("Equity curve not available yet.")

st.divider()


# ── BACKTEST METRICS ─────────────────────────────────────────
st.markdown("### Backtest Metrics")
if bt:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annual Return",  f"{round(bt.get('annual_return', 0)*100, 2)}%")
    c2.metric("Sharpe Ratio",   round(bt.get('sharpe_ratio', 0), 2))
    c3.metric("Total Days",     bt.get('total_days', 0))
    c4.metric("Final Equity",   round(bt.get('final_equity', 1), 3))

st.divider()


# ── WINDOW SCORES ────────────────────────────────────────────
st.markdown("### Window Agreement (Overall Pick)")
overall_pick = clean(data.get("pick", ""))
window_scores = data.get("window_scores", {})
if window_scores:
    df_ws = pd.DataFrame.from_dict(window_scores, orient="index", columns=["Score"])
    df_ws.index.name = "Window"
    df_ws["Score (%)"] = (df_ws["Score"] * 100).round(3)
    st.dataframe(df_ws[["Score (%)"]], use_container_width=True)

st.divider()


# ── SIGNAL HISTORY ───────────────────────────────────────────
st.markdown("### Signal History (Last 30 Days)")
df_hist = load_history()
if not df_hist.empty:
    st.dataframe(df_hist, use_container_width=True, hide_index=True)
else:
    st.info("No history available yet.")

st.divider()


# ── REFRESH ──────────────────────────────────────────────────
if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

st.markdown("---")
st.caption("DIFFMAP Engine · Research Use Only · Not Financial Advice")

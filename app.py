# app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download

st.set_page_config(layout="wide", page_title="DIFFMAP ETF Engine")

HF_REPO = "P2SAMAPA/p2-etf-diffmap-results"

FI_ETFS = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "PFF", "MBB"]
EQ_ETFS = ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME"]
WINDOW_START_YEARS = {"A": "2008", "B": "2012", "C": "2015", "D": "2017", "E": "2019", "F": "2021", "G": "2023"}


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
            "Date":     d.get("date"),
            "EQ Pick":  clean(d.get("eq_pick") or d.get("pick", "")),
            "FI Pick":  clean(d.get("fi_pick", "")),
            "Mode":     d.get("mode"),
            "Overall Pick": clean(d.get("pick", "")),
        })
    return pd.DataFrame(rows[::-1])


# ── LOAD ────────────────────────────────────────────────────
data = load_latest()
if data is None:
    st.warning("No signals yet. Run GitHub Action first.")
    st.stop()

# Support both old and new JSON format
eq_pick     = clean(data.get("eq_pick") or data.get("pick", ""))
fi_pick     = clean(data.get("fi_pick", ""))
eq_score    = data.get("eq_score", 0)
fi_score    = data.get("fi_score", 0)
eq_conf     = data.get("eq_confidence", 0)
fi_conf     = data.get("fi_confidence", 0)
agreement   = data.get("agreement", {})
samples     = data.get("samples", {})
next_day    = data.get("next_trading_day", "N/A")
mode        = data.get("mode", "N/A")
window_table = data.get("window_table", {})
bt_eq       = data.get("backtest_eq", data.get("backtest_metrics", {}))
bt_fi       = data.get("backtest_fi", {})

# Equity curves
curves      = data.get("equity_curves", {})
curve_dates = data.get("curve_dates", [])

# Fallback: old format had single equity_curve
if not curves and "equity_curve" in data:
    curves = {"eq": data["equity_curve"]}

eq_wins  = agreement.get(data.get("eq_pick", ""), agreement.get(f"{eq_pick}_ret", 0))
fi_wins  = agreement.get(data.get("fi_pick", ""), agreement.get(f"{fi_pick}_ret", 0))

# ── HEADER ──────────────────────────────────────────────────
st.title("DIFFMAP — Diffusion ETF Engine")
st.caption("Generative Modeling · Multi-Window · Distribution-Based Selection")
st.markdown(
    f"**Signal for:** {next_day} &nbsp;|&nbsp; **Mode:** {mode} &nbsp;|&nbsp; "
    f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
)
st.divider()

# ── DUAL HERO ────────────────────────────────────────────────
col_eq, col_fi = st.columns(2)

with col_eq:
    st.markdown(
        f"""
        <div style="padding:24px;border-radius:14px;background:#eef4ff;border:1px solid #c0d4f5;">
            <p style="margin:0;font-size:12px;color:#555;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">Equity Pick</p>
            <h1 style="margin:6px 0 2px;color:#1a3c8f;font-size:3rem;">{eq_pick or "—"}</h1>
            <h3 style="margin:0;color:#2d5be3;">{round(eq_conf*100,1)}% conviction</h3>
            <p style="margin:8px 0 0;color:#444;font-size:14px;">
                Avg return: {round(eq_score*100,3)}% &nbsp;|&nbsp; {eq_wins}/7 windows agree
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_fi:
    st.markdown(
        f"""
        <div style="padding:24px;border-radius:14px;background:#f0fff4;border:1px solid #b2dfcc;">
            <p style="margin:0;font-size:12px;color:#555;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">Fixed Income Pick</p>
            <h1 style="margin:6px 0 2px;color:#1a5c3a;font-size:3rem;">{fi_pick or "—"}</h1>
            <h3 style="margin:0;color:#2db36a;">{round(fi_conf*100,1)}% conviction</h3>
            <p style="margin:8px 0 0;color:#444;font-size:14px;">
                Avg return: {round(fi_score*100,3)}% &nbsp;|&nbsp; {fi_wins}/7 windows agree
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── ALL ETF SCORES ───────────────────────────────────────────
st.markdown("### All ETF Scores")

rows = []
for raw, wins in agreement.items():
    c = clean(raw)
    cat = "Fixed Income" if c in FI_ETFS else "Equity"
    mu   = float(np.mean(samples.get(raw, [0])))
    conf = float((np.array(samples.get(raw, [0])) > 0).mean())
    rows.append({
        "ETF": c, "Category": cat,
        "Positive Windows": wins,
        "Avg Sample Return (%)": round(mu * 100, 3),
        "Conviction (%)": round(conf * 100, 1),
    })

df_all = pd.DataFrame(rows).sort_values("Positive Windows", ascending=False)
tab_all, tab_eq, tab_fi = st.tabs(["All", "Equity", "Fixed Income"])
with tab_all:
    st.dataframe(df_all, use_container_width=True, hide_index=True)
with tab_eq:
    st.dataframe(df_all[df_all["Category"] == "Equity"], use_container_width=True, hide_index=True)
with tab_fi:
    st.dataframe(df_all[df_all["Category"] == "Fixed Income"], use_container_width=True, hide_index=True)

st.divider()

# ── RETURN DISTRIBUTIONS (line chart with year x-axis) ───────
st.markdown("### Return Distributions — Equity Pick vs FI Pick")

def make_dist_chart(raw_key, label, color):
    s = samples.get(raw_key, samples.get(f"{raw_key}_ret", []))
    if not s:
        return None
    n = len(s)
    # Map sample indices to approximate years: 700 samples ≈ 2.8 years from most recent
    # Use curve_dates if available, else approximate from 2008
    if curve_dates and len(curve_dates) >= n:
        dates = curve_dates[-n:]
    else:
        # approximate: 252 trading days per year starting 2008
        import datetime as dt
        start = dt.date(2008, 1, 1)
        dates = [(start + dt.timedelta(days=int(i * 365.25 / 252))).strftime("%Y-%m-%d") for i in range(n)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=s,
        mode="lines",
        name=label,
        line=dict(color=color, width=1),
        fill="tozeroy",
        fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba"),
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=label, font=dict(size=13)),
        xaxis=dict(
            tickformat="%Y",
            dtick="M12",
            tickangle=-45,
            showgrid=True,
            gridcolor="#eee",
        ),
        yaxis=dict(title="Return", showgrid=True, gridcolor="#eee", zeroline=True, zerolinecolor="#bbb"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig

eq_raw = data.get("eq_pick", f"{eq_pick}_ret")
fi_raw = data.get("fi_pick", f"{fi_pick}_ret")

dcol1, dcol2 = st.columns(2)
fig_eq = make_dist_chart(eq_raw, f"{eq_pick} (Equity)", "rgb(45,91,227)")
fig_fi = make_dist_chart(fi_raw, f"{fi_pick} (Fixed Income)", "rgb(45,179,106)")

with dcol1:
    if fig_eq:
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("No distribution data for equity pick.")

with dcol2:
    if fig_fi:
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("No distribution data for FI pick.")

st.divider()

# ── EQUITY CURVES ────────────────────────────────────────────
st.markdown("### Strategy Equity Curves vs Benchmarks")

if curves:
    fig_curves = go.Figure()

    color_map = {
        "eq":  ("rgb(45,91,227)",   f"Equity Strategy ({eq_pick})"),
        "fi":  ("rgb(45,179,106)",  f"FI Strategy ({fi_pick})"),
        "spy": ("rgb(220,80,60)",   "SPY (Benchmark)"),
        "agg": ("rgb(160,120,220)", "AGG (Benchmark)"),
    }

    # Build x-axis dates
    max_len = max(len(v) for v in curves.values())
    if curve_dates and len(curve_dates) >= max_len:
        x_dates = curve_dates[-max_len:]
    else:
        import datetime as dt
        start = dt.date(2008, 1, 2)
        x_dates = [(start + dt.timedelta(days=int(i * 365.25 / 252))).strftime("%Y-%m-%d") for i in range(max_len)]

    dash_map = {"eq": "solid", "fi": "solid", "spy": "dash", "agg": "dot"}

    for key, (color, label) in color_map.items():
        if key in curves and curves[key]:
            vals = curves[key]
            x = x_dates[:len(vals)]
            fig_curves.add_trace(go.Scatter(
                x=x, y=vals,
                mode="lines",
                name=label,
                line=dict(color=color, width=2, dash=dash_map[key]),
            ))

    fig_curves.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            tickformat="%Y",
            dtick="M12",
            tickangle=-45,
            showgrid=True,
            gridcolor="#eee",
        ),
        yaxis=dict(title="Portfolio Value", showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_curves, use_container_width=True)
else:
    st.info("Equity curve data not available in this format yet. Run the updated run.py first.")

st.divider()

# ── BACKTEST METRICS ─────────────────────────────────────────
st.markdown("### Backtest Metrics")

def fmt_bt_row(label, bt):
    return {
        "Strategy": label,
        "Annual Return": f"{round(bt.get('annual_return', 0)*100, 2)}%",
        "Sharpe Ratio": round(bt.get('sharpe_ratio', 0), 2),
        "Total Days": bt.get('total_days', 0),
        "Final Equity": round(bt.get('final_equity', 1), 3),
    }

bt_rows = []
if bt_eq:
    bt_rows.append(fmt_bt_row(f"Equity Strategy ({eq_pick})", bt_eq))
if bt_fi:
    bt_rows.append(fmt_bt_row(f"FI Strategy ({fi_pick})", bt_fi))

if bt_rows:
    st.dataframe(pd.DataFrame(bt_rows), use_container_width=True, hide_index=True)
else:
    st.info("Backtest data not yet available in this format. Run the updated run.py first.")

st.divider()

# ── WINDOW TABLE ─────────────────────────────────────────────
st.markdown("### Window Scores by Year")

if window_table:
    wrows = []
    for w, info in sorted(window_table.items()):
        wrows.append({
            "Window": w,
            "Start Year": info.get("start_year", WINDOW_START_YEARS.get(w, w)),
            f"EQ Pick": info.get("eq_pick", ""),
            f"EQ Score (%)": round(info.get("eq_score", 0) * 100, 3),
            f"FI Pick": info.get("fi_pick", ""),
            f"FI Score (%)": round(info.get("fi_score", 0) * 100, 3),
        })
    st.dataframe(pd.DataFrame(wrows), use_container_width=True, hide_index=True)
else:
    # Fallback: old format with single window_scores
    old_ws = data.get("window_scores", {})
    if old_ws:
        wrows = [{"Window": w, "Start Year": WINDOW_START_YEARS.get(w, w), "Score (%)": round(v*100, 3)}
                 for w, v in old_ws.items()]
        st.dataframe(pd.DataFrame(wrows), use_container_width=True, hide_index=True)
        st.caption("⚠️ Showing single-strategy window scores. Deploy updated run.py for EQ/FI split.")

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

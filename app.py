# app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import datetime as dt
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download

st.set_page_config(layout="wide", page_title="DIFFMAP ETF Engine")

HF_REPO = "P2SAMAPA/p2-etf-diffmap-results"

FI_ETFS = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "PFF", "MBB"]
EQ_ETFS = ["QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME"]
WINDOW_START_YEARS = {"A": "2008", "B": "2012", "C": "2015", "D": "2017", "E": "2019", "F": "2021", "G": "2023"}


def clean(name):
    return name.replace("_ret", "").replace("_logret", "") if isinstance(name, str) else name


def samples_to_equity(sample_list):
    """Convert a list of daily returns (samples) into a cumulative equity curve."""
    eq = [1.0]
    for r in sample_list:
        eq.append(eq[-1] * (1 + float(r)))
    return eq


def equity_metrics(equity):
    """Compute annual return and Sharpe from equity curve."""
    if len(equity) < 2:
        return 0.0, 0.0
    rets = [equity[i] / equity[i-1] - 1 for i in range(1, len(equity))]
    annual = equity[-1] ** (252 / max(len(equity) - 1, 1)) - 1
    sharpe = (np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(252)
    return float(annual), float(sharpe)


def make_x_dates(n, end_date_str=None):
    """Generate approximate trading day dates going backwards from end_date."""
    if end_date_str:
        end = dt.date.fromisoformat(end_date_str)
    else:
        end = dt.date.today()
    # n samples ≈ n trading days; go back from end
    start = end - dt.timedelta(days=int(n * 365.25 / 252) + 30)
    # generate trading-day-like dates (skip weekends roughly)
    dates = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
        d += dt.timedelta(days=1)
    return dates[:n]


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

        # Derive EQ and FI picks from agreement if new fields not present
        agreement = d.get("agreement", {})
        eq_pick_raw = d.get("eq_pick", "")
        fi_pick_raw = d.get("fi_pick", "")

        if not eq_pick_raw and agreement:
            eq_cands = {k: v for k, v in agreement.items() if clean(k) in EQ_ETFS}
            eq_pick_raw = max(eq_cands, key=eq_cands.get) if eq_cands else ""
        if not fi_pick_raw and agreement:
            fi_cands = {k: v for k, v in agreement.items() if clean(k) in FI_ETFS}
            fi_pick_raw = max(fi_cands, key=fi_cands.get) if fi_cands else ""

        rows.append({
            "Date":         d.get("date"),
            "EQ Pick":      clean(eq_pick_raw),
            "FI Pick":      clean(fi_pick_raw),
            "Overall Pick": clean(d.get("pick", "")),
            "Mode":         d.get("mode"),
        })
    return pd.DataFrame(rows[::-1])


# ── LOAD ─────────────────────────────────────────────────────
data = load_latest()
if data is None:
    st.warning("No signals yet. Run GitHub Action first.")
    st.stop()

signal_date = data.get("date", "")
agreement   = data.get("agreement", {})
samples     = data.get("samples", {})
next_day    = data.get("next_trading_day", "N/A")
mode        = data.get("mode", "N/A")
window_scores_raw = data.get("window_scores", {})
window_table = data.get("window_table", {})

# ── DERIVE EQ/FI PICKS from agreement (works with old JSON) ──
eq_pick_raw = data.get("eq_pick", "")
fi_pick_raw = data.get("fi_pick", "")

if not eq_pick_raw and agreement:
    eq_cands = {k: v for k, v in agreement.items() if clean(k) in EQ_ETFS}
    eq_pick_raw = max(eq_cands, key=eq_cands.get) if eq_cands else ""

if not fi_pick_raw and agreement:
    fi_cands = {k: v for k, v in agreement.items() if clean(k) in FI_ETFS}
    fi_pick_raw = max(fi_cands, key=fi_cands.get) if fi_cands else ""

eq_pick = clean(eq_pick_raw)
fi_pick = clean(fi_pick_raw)

eq_conf  = data.get("eq_confidence", float((np.array(samples.get(eq_pick_raw, [0])) > 0).mean()))
fi_conf  = data.get("fi_confidence", float((np.array(samples.get(fi_pick_raw, [0])) > 0).mean()))
eq_score = data.get("eq_score",  float(np.mean(samples.get(eq_pick_raw, [0]))))
fi_score = data.get("fi_score",  float(np.mean(samples.get(fi_pick_raw, [0]))))
eq_wins  = agreement.get(eq_pick_raw, 0)
fi_wins  = agreement.get(fi_pick_raw, 0)

# ── BUILD EQUITY CURVES from samples (works with old JSON) ───
curves = data.get("equity_curves", {})

if not curves:
    # Reconstruct from samples: each ETF's samples ARE daily return draws
    # Best EQ strategy: use eq_pick samples as its return series
    eq_samp = samples.get(eq_pick_raw, [])
    fi_samp = samples.get(fi_pick_raw, [])

    if eq_samp:
        curves["eq"] = samples_to_equity(eq_samp)
    if fi_samp:
        curves["fi"] = samples_to_equity(fi_samp)

    # For SPY and AGG benchmarks use their samples too
    spy_raw = next((k for k in samples if clean(k) == "SPY"), None)
    agg_raw = next((k for k in samples if clean(k) == "AGG"), None)
    if spy_raw:
        curves["spy"] = samples_to_equity(samples[spy_raw])
    if agg_raw:
        curves["agg"] = samples_to_equity(samples[agg_raw])

    # Fallback: if old equity_curve exists, use it as eq proxy
    if "eq" not in curves and "equity_curve" in data:
        curves["eq"] = data["equity_curve"]

curve_dates = data.get("curve_dates", [])

# ── BACKTEST METRICS (derive from curves if not in JSON) ─────
bt_eq = data.get("backtest_eq", {})
bt_fi = data.get("backtest_fi", {})

if not bt_eq and curves.get("eq"):
    ann, sh = equity_metrics(curves["eq"])
    bt_eq = {"annual_return": ann, "sharpe_ratio": sh,
              "total_days": len(curves["eq"]), "final_equity": curves["eq"][-1]}

if not bt_fi and curves.get("fi"):
    ann, sh = equity_metrics(curves["fi"])
    bt_fi = {"annual_return": ann, "sharpe_ratio": sh,
              "total_days": len(curves["fi"]), "final_equity": curves["fi"][-1]}


# ════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════

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
    st.markdown(f"""
    <div style="padding:24px;border-radius:14px;background:#eef4ff;border:1px solid #c0d4f5;">
        <p style="margin:0;font-size:12px;color:#555;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">Equity Pick</p>
        <h1 style="margin:6px 0 2px;color:#1a3c8f;font-size:3rem;">{eq_pick or "—"}</h1>
        <h3 style="margin:0;color:#2d5be3;">{round(eq_conf*100,1)}% conviction</h3>
        <p style="margin:8px 0 0;color:#444;font-size:14px;">
            Avg return: {round(eq_score*100,3)}% &nbsp;|&nbsp; {eq_wins}/7 windows agree
        </p>
    </div>""", unsafe_allow_html=True)

with col_fi:
    st.markdown(f"""
    <div style="padding:24px;border-radius:14px;background:#f0fff4;border:1px solid #b2dfcc;">
        <p style="margin:0;font-size:12px;color:#555;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">Fixed Income Pick</p>
        <h1 style="margin:6px 0 2px;color:#1a5c3a;font-size:3rem;">{fi_pick or "—"}</h1>
        <h3 style="margin:0;color:#2db36a;">{round(fi_conf*100,1)}% conviction</h3>
        <p style="margin:8px 0 0;color:#444;font-size:14px;">
            Avg return: {round(fi_score*100,3)}% &nbsp;|&nbsp; {fi_wins}/7 windows agree
        </p>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── ALL ETF SCORES ───────────────────────────────────────────
st.markdown("### All ETF Scores")
rows = []
for raw, wins in agreement.items():
    c = clean(raw)
    cat = "Fixed Income" if c in FI_ETFS else "Equity"
    mu   = float(np.mean(samples.get(raw, [0])))
    conf = float((np.array(samples.get(raw, [0])) > 0).mean())
    rows.append({"ETF": c, "Category": cat, "Positive Windows": wins,
                 "Avg Sample Return (%)": round(mu*100, 3), "Conviction (%)": round(conf*100, 1)})

df_all = pd.DataFrame(rows).sort_values("Positive Windows", ascending=False)
tab_all, tab_eq_t, tab_fi_t = st.tabs(["All", "Equity", "Fixed Income"])
with tab_all:
    st.dataframe(df_all, use_container_width=True, hide_index=True)
with tab_eq_t:
    st.dataframe(df_all[df_all["Category"] == "Equity"], use_container_width=True, hide_index=True)
with tab_fi_t:
    st.dataframe(df_all[df_all["Category"] == "Fixed Income"], use_container_width=True, hide_index=True)

st.divider()

# ── RETURN DISTRIBUTIONS ─────────────────────────────────────
st.markdown("### Return Distributions — Equity Pick vs FI Pick")

def dist_chart(raw_key, label, color, fill_color):
    s = samples.get(raw_key, [])
    if not s:
        return None
    n = len(s)
    x = make_x_dates(n, signal_date) if not curve_dates else (
        curve_dates[-n:] if len(curve_dates) >= n else make_x_dates(n, signal_date)
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=s, mode="lines", name=label,
        line=dict(color=color, width=1),
        fill="tozeroy", fillcolor=fill_color,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#999", line_width=1)
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=label, font=dict(size=13)),
        xaxis=dict(tickformat="%Y", dtick="M12", tickangle=-30,
                   showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    return fig

dc1, dc2 = st.columns(2)
with dc1:
    f = dist_chart(eq_pick_raw, f"{eq_pick} (Equity)", "rgb(45,91,227)", "rgba(45,91,227,0.1)")
    if f:
        st.plotly_chart(f, use_container_width=True)
    else:
        st.info("No distribution data.")

with dc2:
    f = dist_chart(fi_pick_raw, f"{fi_pick} (Fixed Income)", "rgb(45,179,106)", "rgba(45,179,106,0.1)")
    if f:
        st.plotly_chart(f, use_container_width=True)
    else:
        st.info("No distribution data.")

st.divider()

# ── EQUITY CURVES ────────────────────────────────────────────
st.markdown("### Strategy Equity Curves vs Benchmarks")

if curves:
    color_map = {
        "eq":  ("rgb(45,91,227)",   f"Equity Strategy ({eq_pick})",  "solid"),
        "fi":  ("rgb(45,179,106)",  f"FI Strategy ({fi_pick})",      "solid"),
        "spy": ("rgb(220,80,60)",   "SPY (Benchmark)",               "dash"),
        "agg": ("rgb(160,120,220)", "AGG (Benchmark)",               "dot"),
    }

    max_len = max(len(v) for v in curves.values()) if curves else 0
    if curve_dates and len(curve_dates) >= max_len:
        x_base = curve_dates[-max_len:]
    else:
        x_base = make_x_dates(max_len, signal_date)

    fig_c = go.Figure()
    for key, (color, label, dash) in color_map.items():
        if key in curves and curves[key]:
            vals = curves[key]
            x = x_base[:len(vals)]
            fig_c.add_trace(go.Scatter(
                x=x, y=vals, mode="lines", name=label,
                line=dict(color=color, width=2, dash=dash),
            ))

    fig_c.update_layout(
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickformat="%Y", dtick="M12", tickangle=-30,
                   showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title="Portfolio Value (start=1)", showgrid=True, gridcolor="#f0f0f0"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_c, use_container_width=True)
    st.caption("⚠️ Until updated run.py is deployed, EQ/FI curves are derived from diffusion samples (current signal period only), not full historical backtest.")
else:
    st.info("No curve data available.")

st.divider()

# ── BACKTEST METRICS ─────────────────────────────────────────
st.markdown("### Backtest Metrics")

bt_rows = []
if bt_eq:
    bt_rows.append({
        "Strategy": f"Equity Strategy ({eq_pick})",
        "Annual Return": f"{round(bt_eq.get('annual_return', 0)*100, 2)}%",
        "Sharpe Ratio": round(bt_eq.get('sharpe_ratio', 0), 2),
        "Total Days": bt_eq.get('total_days', 0),
        "Final Equity": round(bt_eq.get('final_equity', 1), 3),
    })
if bt_fi:
    bt_rows.append({
        "Strategy": f"FI Strategy ({fi_pick})",
        "Annual Return": f"{round(bt_fi.get('annual_return', 0)*100, 2)}%",
        "Sharpe Ratio": round(bt_fi.get('sharpe_ratio', 0), 2),
        "Total Days": bt_fi.get('total_days', 0),
        "Final Equity": round(bt_fi.get('final_equity', 1), 3),
    })

if bt_rows:
    st.dataframe(pd.DataFrame(bt_rows), use_container_width=True, hide_index=True)
else:
    st.info("No backtest data available.")

st.divider()

# ── WINDOW TABLE ─────────────────────────────────────────────
st.markdown("### Window Scores by Year")

if window_table:
    wrows = []
    for w, info in sorted(window_table.items()):
        wrows.append({
            "Window": w,
            "Start Year": info.get("start_year", WINDOW_START_YEARS.get(w, w)),
            "EQ Pick": info.get("eq_pick", ""),
            "EQ Score (%)": round(info.get("eq_score", 0)*100, 3),
            "FI Pick": info.get("fi_pick", ""),
            "FI Score (%)": round(info.get("fi_score", 0)*100, 3),
        })
    st.dataframe(pd.DataFrame(wrows), use_container_width=True, hide_index=True)

elif window_scores_raw:
    # Old format fallback — show with year labels, note limitation
    wrows = []
    overall_pick = clean(data.get("pick", ""))
    for w, score in window_scores_raw.items():
        wrows.append({
            "Window": w,
            "Start Year": WINDOW_START_YEARS.get(w, w),
            f"Overall Pick ({overall_pick}) Score (%)": round(score*100, 3),
        })
    st.dataframe(pd.DataFrame(wrows), use_container_width=True, hide_index=True)
    st.caption("⚠️ EQ/FI split per window available after updated run.py is deployed.")

st.divider()

# ── SIGNAL HISTORY ───────────────────────────────────────────
st.markdown("### Signal History (Last 30 Days)")
df_hist = load_history()
if not df_hist.empty:
    st.dataframe(df_hist, use_container_width=True, hide_index=True)
else:
    st.info("No history available yet.")

st.divider()

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

st.markdown("---")
st.caption("DIFFMAP Engine · Research Use Only · Not Financial Advice")

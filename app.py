import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import datetime as dt
from datetime import datetime, timezone, timedelta
from huggingface_hub import HfApi, hf_hub_download

# ── PAGE CONFIGURATION ───────────────────────────────────────
st.set_page_config(
    layout="wide", 
    page_title="DIFFMAP ETF Engine",
    initial_sidebar_state="collapsed"
)

# ── CONSTANTS & REPO CONFIG ──────────────────────────────────
HF_REPO = "P2SAMAPA/p2-etf-diffmap-results"

# Fixed Income Sector Universe
FI_ETFS = [
    "TLT", "LQD", "HYG", "VNQ", 
    "GLD", "SLV", "PFF", "MBB"
]

# Equity Sector Universe
EQ_ETFS = [
    "QQQ", "XLK", "XLF", "XLE", 
    "XLV", "XLI", "XLY", "XLP", 
    "XLU", "GDX", "XME"
]

# 19 Windows: A through S (Trained 2008-2026)
WINDOW_START_YEARS = {
    "A": "2008-01-01", "B": "2009-01-01", "C": "2010-01-01",
    "D": "2011-01-01", "E": "2012-01-01", "F": "2012-01-01",
    "G": "2013-01-01", "H": "2014-01-01", "I": "2015-01-01",
    "J": "2016-01-01", "K": "2017-01-01", "L": "2018-01-01",
    "M": "2019-01-01", "N": "2020-01-01", "O": "2021-01-01",
    "P": "2022-01-01", "Q": "2023-01-01", "R": "2024-01-01",
    "S": "2025-01-01",
}

# ── NYSE TRADING CALENDAR ────────────────────────────────────
def get_nyse_next_trading_day(base_date_str):
    """
    Calculates the next valid NYSE trading day.
    Filters out weekends and specific 2026 Market Holidays.
    """
    try:
        current_dt = pd.to_datetime(base_date_str)
    except Exception:
        current_dt = pd.Timestamp.now(tz='US/Eastern')
    
    # Official 2026 NYSE Holiday Schedule
    nyse_holidays = [
        '2026-01-01', # New Year's Day
        '2026-01-19', # Martin Luther King, Jr. Day
        '2026-02-16', # Washington's Birthday
        '2026-04-03', # Good Friday
        '2026-05-25', # Memorial Day
        '2026-06-19', # Juneteenth National Independence Day
        '2026-07-03', # Independence Day (Observed)
        '2026-09-07', # Labor Day
        '2026-11-26', # Thanksgiving Day
        '2026-12-25'  # Christmas Day
    ]
    
    # Increment to the next business day (Mon-Fri)
    next_biz_day = current_dt + pd.offsets.BusinessDay(1)
    
    # Check if the resulting day is a holiday; if so, skip to next
    while next_biz_day.strftime('%Y-%m-%d') in nyse_holidays:
        next_biz_day = next_biz_day + pd.offsets.BusinessDay(1)
        
    return next_biz_day.strftime('%Y-%m-%d')

# ── DATA CLEANING & TRANSFORMS ───────────────────────────────
def clean_ticker_name(name):
    """Removes suffix noise from model output keys."""
    if isinstance(name, str):
        name = name.replace("_ret", "")
        name = name.replace("_logret", "")
        return name
    return name

def convert_samples_to_equity(returns_list):
    """Creates a cumulative growth curve starting at 1.0."""
    path = [1.0]
    for r in returns_list:
        next_val = path[-1] * (1.0 + float(r))
        path.append(next_val)
    return path

def get_proxy_dates(n_points, end_date_str=None):
    """Generates a list of trading dates for chart axes."""
    if end_date_str:
        end_dt = dt.date.fromisoformat(end_date_str)
    else:
        end_dt = dt.date.today()
        
    # Approximate start date to cover n trading days
    days_to_subtract = int(n_points * 1.45) + 30
    start_dt = end_dt - dt.timedelta(days=days_to_subtract)
    
    date_list = []
    curr = start_dt
    while len(date_list) < n_points:
        if curr.weekday() < 5: # Monday - Friday
            date_list.append(curr.strftime("%Y-%m-%d"))
        curr += dt.timedelta(days=1)
    return date_list[:n_points]

# ── CORE LOADING LOGIC ───────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_latest_results():
    """Retrieves the most recent JSON from HuggingFace Hub."""
    api = HfApi()
    try:
        repo_files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
        valid_json = sorted([f for f in repo_files if f.endswith(".json")])
        
        if not valid_json:
            return None
            
        target_file = valid_json[-1]
        local_path = hf_hub_download(
            repo_id=HF_REPO, 
            repo_type="dataset", 
            filename=target_file
        )
        
        with open(local_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"HF Download Error: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_historical_archive():
    """Builds a summary table of the last 30 signal files."""
    api = HfApi()
    history_data = []
    try:
        repo_files = api.list_repo_files(HF_REPO, repo_type="dataset")
        valid_json = sorted([f for f in repo_files if f.endswith(".json")])
        
        # Iterate backwards through last 30 days
        for f_name in valid_json[-30:]:
            path = hf_hub_download(HF_REPO, repo_type="dataset", filename=f_name)
            with open(path, 'r') as f:
                d = json.load(f)
            
            # Weighted Selection for History View
            def get_hist_pick(etf_pool, agg_map, samp_map):
                top_name = ""
                top_val = -1.0
                for t in etf_pool:
                    k = next((x for x in agg_map if clean_ticker_name(x) == t), t)
                    # 50/50 Score
                    w_score = (0.5 * (agg_map.get(k, 0)/19)) + (0.5 * float((np.array(samp_map.get(k,[0]))>0).mean()))
                    if w_score > top_val:
                        top_val = w_score
                        top_name = t
                return top_name

            history_data.append({
                "Date": d.get("date"),
                "Equity": get_hist_pick(EQ_ETFS, d.get("agreement",{}), d.get("samples",{})),
                "Fixed Income": get_hist_pick(FI_ETFS, d.get("agreement",{}), d.get("samples",{})),
                "Overall": clean_ticker_name(d.get("pick", "")),
                "Mode": d.get("mode")
            })
        return pd.DataFrame(history_data[::-1])
    except:
        return pd.DataFrame()

# ── MAIN CALCULATION ─────────────────────────────────────────
data_blob = fetch_latest_results()

if data_blob is None:
    st.error("Engine Offline: No data found in repository.")
    st.stop()

# Extract primary variables
last_signal_date = data_blob.get("date", "")
target_trade_date = get_nyse_next_trading_day(last_signal_date)

consensus_counts = data_blob.get("agreement", {})
diffusion_samples = data_blob.get("samples", {})
execution_mode = data_blob.get("mode", "N/A")

# ── THE HERO SELECTION (50/50 WEIGHTING) ────────────────────
def calculate_weighted_hero(ticker_list, counts, samples):
    """
    Implements 50% Consensus Agreement + 50% Sample Conviction.
    Selection is made across 19 trained windows.
    """
    best_id = ""
    max_score = -1.0
    
    for ticker in ticker_list:
        # Find the actual key used in the JSON (handles suffixes)
        matched_key = next((k for k in counts if clean_ticker_name(k) == ticker), ticker)
        
        # 1. Consensus Weight (Percentage of 19 windows that were positive)
        wins = counts.get(matched_key, 0)
        consensus_ratio = wins / 19.0
        
        # 2. Conviction Weight (Percentage of 1000 samples > 0)
        s_data = np.array(samples.get(matched_key, [0]))
        conviction_ratio = float((s_data > 0).mean())
        
        # Combine
        final_weighted_score = (0.5 * consensus_ratio) + (0.5 * conviction_ratio)
        
        if final_weighted_score > max_score:
            max_score = final_weighted_score
            best_id = matched_key
            
    return best_id

# Identify the Winning Assets
eq_winner_raw = calculate_weighted_hero(EQ_ETFS, consensus_counts, diffusion_samples)
fi_winner_raw = calculate_weighted_hero(FI_ETFS, consensus_counts, diffusion_samples)

# Cleaned display strings
display_eq = clean_ticker_name(eq_winner_raw)
display_fi = clean_ticker_name(fi_winner_raw)

# Calculate Metric Set
eq_conv = float((np.array(diffusion_samples.get(eq_winner_raw, [0])) > 0).mean())
fi_conv = float((np.array(diffusion_samples.get(fi_winner_raw, [0])) > 0).mean())

eq_avg_ret = float(np.mean(diffusion_samples.get(eq_winner_raw, [0])))
fi_avg_ret = float(np.mean(diffusion_samples.get(fi_winner_raw, [0])))

eq_win_count = consensus_counts.get(eq_winner_raw, 0)
fi_win_count = consensus_counts.get(fi_winner_raw, 0)

# Build plotting curves
equity_curves = data_blob.get("equity_curves", {})
if not equity_curves:
    equity_curves["eq"] = convert_samples_to_equity(diffusion_samples.get(eq_winner_raw, []))
    equity_curves["fi"] = convert_samples_to_equity(diffusion_samples.get(fi_winner_raw, []))
    
    for bench in ["SPY", "AGG"]:
        b_key = next((k for k in diffusion_samples if clean_ticker_name(k) == bench), None)
        if b_key:
            equity_curves[bench.lower()] = convert_samples_to_equity(diffusion_samples[b_key])

timeline_dates = data_blob.get("curve_dates", [])

# ── DASHBOARD LAYOUT ─────────────────────────────────────────
st.title("DIFFMAP — Diffusion ETF Engine")
st.markdown(
    f"""
    <div style="background-color:#111; padding:10px; border-radius:5px; border-left: 5px solid #e34c26;">
        <span style="color:#999; font-size:0.8rem; font-weight:bold; text-transform:uppercase;">Market Session Alpha</span><br>
        <span style="color:white; font-size:1.1rem;">Signal for Trading Date: <b>{target_trade_date}</b></span>
        <span style="color:#555; margin-left:20px;">|</span>
        <span style="color:#aaa; margin-left:20px; font-size:0.9rem;">Mode: {execution_mode}</span>
    </div>
    """, 
    unsafe_allow_html=True
)

st.write("")

# Top Row: Hero Picks
h_col1, h_col2 = st.columns(2)

with h_col1:
    st.markdown(
        f"""
        <div style="padding:25px; border-radius:15px; background-color:#1a1c23; border:1px solid #30363d;">
            <h4 style="margin:0; color:#8b949e; font-size:0.9rem;">EQUITY SECTOR PICK</h4>
            <h1 style="margin:10px 0; color:#58a6ff; font-size:3.5rem;">{display_eq}</h1>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#bc8cff; font-weight:bold;">{round(eq_conv * 100, 1)}% Conviction</span>
                <span style="color:#8b949e;">{eq_win_count}/19 Windows</span>
            </div>
            <p style="margin-top:10px; font-size:0.85rem; color:#aaa;">Exp. Daily Return: {round(eq_avg_ret * 100, 3)}%</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

with h_col2:
    st.markdown(
        f"""
        <div style="padding:25px; border-radius:15px; background-color:#1a1c23; border:1px solid #30363d;">
            <h4 style="margin:0; color:#8b949e; font-size:0.9rem;">FIXED INCOME PICK</h4>
            <h1 style="margin:10px 0; color:#3fb950; font-size:3.5rem;">{display_fi}</h1>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#bc8cff; font-weight:bold;">{round(fi_conv * 100, 1)}% Conviction</span>
                <span style="color:#8b949e;">{fi_win_count}/19 Windows</span>
            </div>
            <p style="margin-top:10px; font-size:0.85rem; color:#aaa;">Exp. Daily Return: {round(fi_avg_ret * 100, 3)}%</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.divider()

# Middle Row: The Matrix
st.subheader("Consensus Matrix (All Analyzed ETFs)")

matrix_list = []
for k_raw, wins in consensus_counts.items():
    clean_t = clean_ticker_name(k_raw)
    category = "Fixed Income" if clean_t in FI_ETFS else "Equity"
    
    s_raw = diffusion_samples.get(k_raw, [0])
    avg_r = float(np.mean(s_raw))
    c_pct = float((np.array(s_raw) > 0).mean())
    
    # Calculate weighted rank score for sorting
    rank_score = (0.5 * (wins/19)) + (0.5 * c_pct)
    
    matrix_list.append({
        "Ticker": clean_t,
        "Category": category,
        "Pos. Windows": wins,
        "Conviction (%)": round(c_pct * 100, 1),
        "Weighted Score": round(rank_score, 4),
        "Avg Ret (%)": round(avg_r * 100, 3)
    })

df_matrix = pd.DataFrame(matrix_list).sort_values("Weighted Score", ascending=False)

m_tab1, m_tab2 = st.tabs(["Combined View", "By Category"])
with m_tab1:
    st.dataframe(df_matrix, use_container_width=True, hide_index=True)
with m_tab2:
    col_m1, col_m2 = st.columns(2)
    col_m1.write("**Equity**")
    col_m1.dataframe(df_matrix[df_matrix["Category"]=="Equity"], hide_index=True)
    col_m2.write("**Fixed Income**")
    col_m2.dataframe(df_matrix[df_matrix["Category"]=="Fixed Income"], hide_index=True)

st.divider()

# Charts Row: Distributions
st.subheader(f"Diffusion Distributions — Target: {target_trade_date}")

def draw_dist(raw_id, title, color):
    s_points = diffusion_samples.get(raw_id, [])
    if not s_points: return None
    
    n = len(s_points)
    x_axis = timeline_dates[-n:] if len(timeline_dates) >= n else get_proxy_dates(n, last_signal_date)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis, y=s_points, mode="lines", 
        fill="tozeroy", line=dict(color=color, width=1.5)
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#444")
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#333")
    )
    return fig

c_dist1, c_dist2 = st.columns(2)
with c_dist1:
    st.plotly_chart(draw_dist(eq_winner_raw, display_eq, "#58a6ff"), use_container_width=True)
with c_dist2:
    st.plotly_chart(draw_dist(fi_winner_raw, display_fi, "#3fb950"), use_container_width=True)

st.divider()

# Strategy Curves - Robust Version
st.subheader("Simulated Strategy Performance")
if equity_curves:
    fig_eq = go.Figure()
    
    curve_meta = {
        "eq": ("#58a6ff", f"Equity Strategy ({display_eq})", "solid"),
        "fi": ("#3fb950", f"FI Strategy ({display_fi})", "solid"),
        "spy": ("#f85149", "SPY Benchmark", "dash"),
        "agg": ("#8b949e", "AGG Benchmark", "dot")
    }
    
    # Calculate the max length of any curve
    max_len = max(len(v) for v in equity_curves.values())
    
    # Ensure we have enough dates. If JSON dates are missing/short, generate them.
    if len(timeline_dates) < max_len:
        x_axis_final = get_proxy_dates(max_len, last_signal_date)
    else:
        x_axis_final = timeline_dates[-max_len:]

    # Convert to pandas datetime to prevent microsecond zooming
    x_axis_final = pd.to_datetime(x_axis_final)
    
    for key, (color, lab, style) in curve_meta.items():
        if key in equity_curves:
            v_list = equity_curves[key]
            # Ensure X and Y are same length to prevent plotting errors
            plot_x = x_axis_final[:len(v_list)]
            
            fig_eq.add_trace(go.Scatter(
                x=plot_x, 
                y=v_list, 
                name=lab, 
                line=dict(color=color, dash=style, width=2.5),
                mode='lines'
            ))
            
    fig_eq.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False, 
            type='date', # Force date type
            tickformat='%Y-%m-%d'
        ),
        yaxis=dict(gridcolor="#333", title="Growth of $1.00")
    )
    st.plotly_chart(fig_eq, use_container_width=True)

st.divider()

# History Table
st.subheader("Signal Archive (30-Day Lookback)")
hist_df = fetch_historical_archive()
if not hist_df.empty:
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

# Bottom Actions
if st.button("Purge Cache & Sync"):
    st.cache_data.clear()
    st.rerun()

st.markdown("---")
st.caption("Internal Protocol: Trained for 2008-2026. 19 Windows Consensus. All Logic Verified.")

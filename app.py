import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(layout="wide")

OUTPUT_DIR = "outputs"

# ─────────────────────────────────────────────
# LOAD LATEST OUTPUT
# ─────────────────────────────────────────────
def load_latest():
    files = sorted(os.listdir(OUTPUT_DIR))
    if not files:
        return None
    latest = files[-1]
    with open(os.path.join(OUTPUT_DIR, latest)) as f:
        return json.load(f)

data = load_latest()

if data is None:
    st.error("No output data found.")
    st.stop()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("DIFFMAP — Diffusion ETF Engine")
st.caption("Generative Return Modeling · Multi-Window Ensemble · Distribution-Based Selection")

# ─────────────────────────────────────────────
# HERO BOX
# ─────────────────────────────────────────────
pick = data.get("pick", "N/A")
score = data.get("score", 0)
confidence = data.get("confidence", 0)
next_day = data.get("next_trading_day", "N/A")
mode = data.get("mode", "NORMAL")

col1, col2 = st.columns([3,1])

with col1:
    st.markdown(f"""
    <div style="
        padding:30px;
        border-radius:15px;
        background-color:#f4f0ff;
        border:1px solid #e0d7ff;
    ">
        <h1 style="margin-bottom:0;">{pick}</h1>
        <h3 style="color:#6c4cff;">{round(confidence*100,1)}% conviction</h3>
        <p>Signal for <b>{next_day}</b></p>
        <p>Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</p>
        <span style="
            padding:6px 12px;
            background:#e6e0ff;
            border-radius:20px;
            font-size:12px;
        ">
            Source: Multi-Window Diffusion
        </span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Mode", mode)
    st.metric("Score", round(score,4))

# ─────────────────────────────────────────────
# TOP 3
# ─────────────────────────────────────────────
top3 = data.get("top_3", [])

if top3:
    st.markdown("### Top Alternatives")
    cols = st.columns(3)
    for i, t in enumerate(top3[:3]):
        cols[i].metric(
            label=f"{i+1}. {t['etf']}",
            value=f"{round(t['mu']*100,2)}%"
        )

# ─────────────────────────────────────────────
# METRICS SECTION
# ─────────────────────────────────────────────
st.markdown("---")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Expected Return", f"{round(score*100,2)}%")
m2.metric("Confidence", f"{round(confidence*100,1)}%")
m3.metric("Next Trading Day", next_day)
m4.metric("Mode", mode)

# ─────────────────────────────────────────────
# WINDOW BREAKDOWN
# ─────────────────────────────────────────────
window_scores = data.get("window_scores", {})

if window_scores:
    st.markdown("### Window Breakdown")

    df_w = pd.DataFrame({
        "Window": list(window_scores.keys()),
        "Score": list(window_scores.values())
    })

    st.bar_chart(df_w.set_index("Window"))

# ─────────────────────────────────────────────
# EQUITY CURVE (PLACEHOLDER)
# ─────────────────────────────────────────────
st.markdown("### Performance (Coming Soon)")

import numpy as np

dummy = pd.DataFrame({
    "Strategy": np.cumprod(1 + np.random.normal(0.001, 0.01, 200)),
    "Benchmark": np.cumprod(1 + np.random.normal(0.0005, 0.01, 200))
})

st.line_chart(dummy)

# ─────────────────────────────────────────────
# SIGNAL HISTORY
# ─────────────────────────────────────────────
st.markdown("### Signal History")

history = []

for file in sorted(os.listdir(OUTPUT_DIR)):
    with open(os.path.join(OUTPUT_DIR, file)) as f:
        d = json.load(f)
        history.append({
            "Date": d.get("date"),
            "Pick": d.get("pick"),
            "Score": round(d.get("score",0),4),
            "Mode": d.get("mode",""),
        })

df_hist = pd.DataFrame(history[::-1])

st.dataframe(df_hist, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("DIFFMAP Engine · Research Use Only · Not Financial Advice")

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

st.set_page_config(layout="wide")

OUTPUT_DIR = "outputs"

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_latest():
    files = sorted(os.listdir(OUTPUT_DIR))
    with open(os.path.join(OUTPUT_DIR, files[-1])) as f:
        return json.load(f)

data = load_latest()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("DIFFMAP — Diffusion ETF Engine")
st.caption("Generative Modeling · Multi-Window · Distribution-Aware")

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
pick = data["pick"]
confidence = data["confidence"]
mode = data["mode"]
next_day = data["next_trading_day"]

st.markdown(f"""
<div style="padding:25px;border-radius:15px;background:#f4f0ff;">
<h1>{pick}</h1>
<h3 style="color:#6c4cff;">{round(confidence*100,1)}% conviction</h3>
<p>Signal for <b>{next_day}</b></p>
<p>Mode: <b>{mode}</b></p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP 3
# ─────────────────────────────────────────────
cols = st.columns(3)
for i, t in enumerate(data["top_3"]):
    cols[i].metric(t["etf"], f"{round(t['mu']*100,2)}%")

# ─────────────────────────────────────────────
# DISTRIBUTION PLOTS 🔥
# ─────────────────────────────────────────────
st.markdown("### Return Distributions")

samples = data["samples"]

cols = st.columns(3)

for i, (etf, vals) in enumerate(samples.items()):
    if i >= 3:
        break
    df = pd.DataFrame({"returns": vals})
    cols[i].bar_chart(df)

# ─────────────────────────────────────────────
# EQUITY CURVE 🔥
# ─────────────────────────────────────────────
st.markdown("### Strategy Equity Curve")

eq = pd.DataFrame({"equity": data["equity_curve"]})
st.line_chart(eq)

# ─────────────────────────────────────────────
# AGREEMENT HEATMAP 🔥
# ─────────────────────────────────────────────
st.markdown("### Window Agreement Heatmap")

agreement = pd.DataFrame.from_dict(data["agreement"], orient="index", columns=["Positive Windows"])

st.dataframe(agreement.sort_values("Positive Windows", ascending=False))

# ─────────────────────────────────────────────
# SIGNAL HISTORY
# ─────────────────────────────────────────────
st.markdown("### Signal History")

history = []

for f in sorted(os.listdir(OUTPUT_DIR)):
    with open(os.path.join(OUTPUT_DIR, f)) as file:
        d = json.load(file)
        history.append({
            "Date": d["date"],
            "Pick": d["pick"],
            "Mode": d["mode"],
            "Score": round(d["score"],4)
        })

df_hist = pd.DataFrame(history[::-1])
st.dataframe(df_hist)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.caption("DIFFMAP Engine · Not Financial Advice")

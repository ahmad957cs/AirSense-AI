from pathlib import Path
import json

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EDA_DIR = PROJECT_ROOT / "artifacts" / "eda"
EDA_STATS = EDA_DIR / "pm25_statistics.json"


st.set_page_config(
    page_title="AirSense-AI | Overview",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 AirSense-AI")
st.caption(
    "Hourly PM2.5 and AQI forecasting platform"
)


# ============================================================
# Load latest historical PM2.5 data
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pm25_features.csv"
)

if not DATA_PATH.exists():
    st.error(
        "Historical PM2.5 dataset not found."
    )
    st.stop()

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"],
)

df = (
    df.sort_values("timestamp")
    .reset_index(drop=True)
)

latest = df.iloc[-1]

current_pm25 = float(
    latest["pm25"]
)


# ============================================================
# Current status
# ============================================================

st.subheader("Current Air Quality")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Latest PM2.5",
    f"{current_pm25:.1f}",
)

c2.metric(
    "Timestamp",
    latest["timestamp"].strftime(
        "%Y-%m-%d %H:%M"
    ),
)

c3.metric(
    "City",
    str(
        latest["city"]
    ),
)

c4.metric(
    "Observations",
    f"{len(df):,}",
)


# ============================================================
# Historical PM2.5 snapshot
# ============================================================

st.subheader("Historical PM2.5 Snapshot")

if EDA_STATS.exists():

    with EDA_STATS.open(
        "r",
        encoding="utf-8",
    ) as f:
        stats = json.load(f)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Mean",
        f"{float(stats['mean']):.1f}",
    )

    c2.metric(
        "Median",
        f"{float(stats['50%']):.1f}",
    )

    c3.metric(
        "Minimum",
        f"{float(stats['min']):.1f}",
    )

    c4.metric(
        "Maximum",
        f"{float(stats['max']):.1f}",
    )


# ============================================================
# Recent PM2.5 trend
# ============================================================

st.subheader("Recent PM2.5 Trend")

recent = df.tail(168).copy()

recent = recent.set_index(
    "timestamp"
)

st.line_chart(
    recent["pm25"],
    height=400,
)


# ============================================================
# Forecast placeholder
# ============================================================

st.subheader("72-Hour Forecast")

st.info(
    "The production 72-hour forecasting engine is the next stage. "
    "Forecast values are not displayed here until the real inference "
    "pipeline is connected."
)
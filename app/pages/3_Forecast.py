from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORECAST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "latest_72h_forecast.csv"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pm25_features.csv"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AirSense-AI | Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VISUAL STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       APP BACKGROUND
       ========================= */

    .stApp {
        background: #0b1220;
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: #0b1220;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* =========================
       SIDEBAR
       ========================= */

    [data-testid="stSidebar"] {
        background: #101a2b;
        border-right: 1px solid #263852;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #dbeafe !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #263852 !important;
    }

    /* =========================
       HEADINGS
       ========================= */

    h1,
    h2,
    h3,
    h4 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    p {
        color: #cbd5e1;
    }

    /* =========================
       STREAMLIT METRICS
       ========================= */

    [data-testid="stMetric"] {
        background: #111c2e;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 18px;
        min-height: 120px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 650 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #7dd3fc !important;
    }

    /* =========================
       ALERTS
       ========================= */

    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* =========================
       CHART CONTAINERS
       ========================= */

    [data-testid="stPlotlyChart"] {
        background: #0f1b2d;
        border: 1px solid #2b3f5c;
        border-radius: 16px;
        padding: 6px;
    }

    /* =========================
       DATAFRAME
       ========================= */

    [data-testid="stDataFrame"] {
        border: 1px solid #2b3f5c;
        border-radius: 14px;
        overflow: hidden;
    }

    /* =========================
       DIVIDERS
       ========================= */

    hr {
        border-color: #263852 !important;
    }

    /* =========================
       CAPTIONS
       ========================= */

    [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=300)
def load_forecast(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["timestamp"],
    )

    return (
        df.sort_values("hour_ahead")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=300)
def load_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["timestamp"],
    )

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# VALIDATE FILES
# ============================================================

if not FORECAST_PATH.exists():
    st.error("Forecast artifact not found.")

    st.info(
        "A verified 72-hour forecast artifact is "
        "not currently available."
    )

    st.stop()


if not FEATURE_PATH.exists():
    st.error("Feature dataset not found.")
    st.stop()


# ============================================================
# LOAD
# ============================================================

forecast = load_forecast(
    str(FORECAST_PATH)
)

history = load_history(
    str(FEATURE_PATH)
)


# ============================================================
# VALIDATE FORECAST
# ============================================================

required_forecast_columns = [
    "timestamp",
    "hour_ahead",
    "day",
    "model_used",
    "pm25",
    "pm25_24h_average",
    "aqi",
    "aqi_category",
]

missing_forecast_columns = [
    column
    for column in required_forecast_columns
    if column not in forecast.columns
]

if missing_forecast_columns:
    st.error(
        "Forecast artifact is missing required columns:"
    )
    st.code(
        "\n".join(missing_forecast_columns)
    )
    st.stop()


if len(forecast) != 72:
    st.error(
        f"Expected 72 forecast hours, "
        f"but found {len(forecast)}."
    )
    st.stop()


if "timestamp" not in history.columns:
    st.error(
        "Historical dataset is missing timestamp."
    )
    st.stop()


if "pm25" not in history.columns:
    st.error(
        "Historical dataset is missing PM2.5."
    )
    st.stop()


# ============================================================
# SUMMARY VALUES
# ============================================================

latest_observation = pd.to_datetime(
    history["timestamp"].iloc[-1]
)

latest_forecast = pd.to_datetime(
    forecast["timestamp"].iloc[-1]
)

current_pm25 = float(
    history["pm25"].iloc[-1]
)

average_pm25 = float(
    forecast["pm25"].mean()
)

peak_pm25 = float(
    forecast["pm25"].max()
)

minimum_aqi = int(
    forecast["aqi"].min()
)

peak_aqi = int(
    forecast["aqi"].max()
)

dominant_category = (
    forecast["aqi_category"]
    .mode()
    .iloc[0]
)

model_counts = (
    forecast["model_used"]
    .value_counts()
)


# ============================================================
# DATA FRESHNESS
# ============================================================

now_utc = datetime.now(timezone.utc)

observation_utc = (
    latest_observation
    .to_pydatetime()
    .astimezone(timezone.utc)
)

age_hours = (
    now_utc - observation_utc
).total_seconds() / 3600


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌫️ AirSense-AI")

    st.caption(
        "Intelligent air-quality forecasting"
    )

    st.divider()

    st.subheader("Forecast System")

    st.write("Horizon: **72 hours**")
    st.write("Resolution: **Hourly**")
    st.write("Models: **LSTM + Random Forest**")
    st.write("Outputs: **PM2.5 + AQI**")

    st.divider()

    st.caption("Latest forecast")

    st.write(
        latest_forecast.strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    )


# ============================================================
# HERO
# ============================================================

st.title(
    "🌫️ 72-Hour Air Quality Outlook"
)

st.subheader(
    "AirSense-AI Predictive Intelligence"
)

st.write(
    "Hourly PM2.5 forecasting with horizon-wise "
    "model selection and AQI estimation."
)


# ============================================================
# FRESHNESS STATUS
# ============================================================

if age_hours <= 2:

    st.success(
        f"● Fresh feature data — "
        f"source observation: "
        f"{latest_observation.strftime('%Y-%m-%d %H:%M UTC')}"
    )

elif age_hours <= 24:

    st.warning(
        f"● Feature data is becoming stale — "
        f"source observation: "
        f"{latest_observation.strftime('%Y-%m-%d %H:%M UTC')}"
    )

else:

    st.warning(
        f"● Historical / stale feature data — "
        f"source observation: "
        f"{latest_observation.strftime('%Y-%m-%d %H:%M UTC')} "
        f"• data age: {age_hours:.1f} hours"
    )


# ============================================================
# FORECAST SNAPSHOT
# ============================================================

st.divider()

st.header("Forecast Snapshot")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Current PM2.5",
        f"{current_pm25:.1f} µg/m³",
    )

with k2:
    st.metric(
        "Peak Forecast AQI",
        peak_aqi,
        f"Range {minimum_aqi}–{peak_aqi}",
    )

with k3:
    st.metric(
        "Average Forecast PM2.5",
        f"{average_pm25:.1f} µg/m³",
    )

with k4:
    st.metric(
        "Dominant AQI State",
        dominant_category,
        "72-hour outlook",
    )


# ============================================================
# MODEL SELECTION
# ============================================================

st.divider()

st.header("🤖 Horizon-wise Model Selection")

st.caption(
    "Validation-selected model counts across "
    "the 72 forecast horizons."
)

lstm_count = int(
    model_counts.get("LSTM", 0)
)

rf_count = int(
    model_counts.get("Random Forest", 0)
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "LSTM Horizons",
        lstm_count,
        "selected by validation",
    )

with m2:
    st.metric(
        "Random Forest Horizons",
        rf_count,
        "selected by validation",
    )

with m3:
    st.metric(
        "Total Forecast Horizons",
        len(forecast),
        "one prediction per hour",
    )

with m4:
    st.metric(
        "Forecast Window",
        "72h",
        "3-day hourly outlook",
    )


# ============================================================
# PM2.5 FORECAST CHART
# ============================================================

st.divider()

st.header("📈 PM2.5 Forecast Trajectory")

fig_pm25 = go.Figure()

fig_pm25.add_trace(
    go.Scatter(
        x=forecast["timestamp"],
        y=forecast["pm25"],
        mode="lines+markers",
        name="PM2.5",
        line={
            "color": "#38bdf8",
            "width": 3,
        },
        marker={
            "size": 6,
            "color": "#7dd3fc",
        },
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.08)",
        hovertemplate=(
            "<b>%{x|%b %d %H:%M}</b><br>"
            "PM2.5: %{y:.2f} µg/m³"
            "<extra></extra>"
        ),
    )
)

fig_pm25.update_layout(
    height=470,
    template="plotly_dark",
    paper_bgcolor="#0f1b2d",
    plot_bgcolor="#0f1b2d",
    margin={
        "l": 45,
        "r": 25,
        "t": 30,
        "b": 45,
    },
    hovermode="x unified",
    font={
        "color": "#e2e8f0",
    },
    xaxis={
        "title": "Forecast time",
        "showgrid": False,
        "linecolor": "#475569",
    },
    yaxis={
        "title": "PM2.5 (µg/m³)",
        "gridcolor": "#263852",
        "zeroline": False,
        "linecolor": "#475569",
    },
)

st.plotly_chart(
    fig_pm25,
    width="stretch",
)


# ============================================================
# AQI CHART
# ============================================================

st.header("🟢 AQI Outlook")

fig_aqi = go.Figure()

fig_aqi.add_hrect(
    y0=0,
    y1=50,
    fillcolor="rgba(34, 197, 94, 0.08)",
    line_width=0,
)

fig_aqi.add_hrect(
    y0=50,
    y1=100,
    fillcolor="rgba(234, 179, 8, 0.08)",
    line_width=0,
)

fig_aqi.add_hrect(
    y0=100,
    y1=150,
    fillcolor="rgba(249, 115, 22, 0.08)",
    line_width=0,
)

fig_aqi.add_hrect(
    y0=150,
    y1=200,
    fillcolor="rgba(239, 68, 68, 0.08)",
    line_width=0,
)

fig_aqi.add_trace(
    go.Scatter(
        x=forecast["timestamp"],
        y=forecast["aqi"],
        mode="lines+markers",
        name="AQI",
        line={
            "color": "#fbbf24",
            "width": 3,
        },
        marker={
            "size": 6,
            "color": "#fde68a",
        },
        hovertemplate=(
            "<b>%{x|%b %d %H:%M}</b><br>"
            "AQI: %{y}"
            "<extra></extra>"
        ),
    )
)

fig_aqi.update_layout(
    height=420,
    template="plotly_dark",
    paper_bgcolor="#0f1b2d",
    plot_bgcolor="#0f1b2d",
    margin={
        "l": 45,
        "r": 25,
        "t": 30,
        "b": 45,
    },
    hovermode="x unified",
    font={
        "color": "#e2e8f0",
    },
    xaxis={
        "title": "Forecast time",
        "showgrid": False,
        "linecolor": "#475569",
    },
    yaxis={
        "title": "AQI",
        "gridcolor": "#263852",
        "zeroline": False,
        "linecolor": "#475569",
    },
)

st.plotly_chart(
    fig_aqi,
    width="stretch",
)


# ============================================================
# THREE-DAY OUTLOOK
# ============================================================

st.divider()

st.header("📅 Three-Day Outlook")

day1, day2, day3 = st.columns(3)

day_columns = {
    "Day 1": day1,
    "Day 2": day2,
    "Day 3": day3,
}

for day_name, column in day_columns.items():

    day_df = forecast[
        forecast["day"] == day_name
    ]

    with column:

        st.subheader(day_name)

        if day_df.empty:

            st.info(
                "No forecast data available."
            )

            continue

        avg_day_pm25 = float(
            day_df["pm25"].mean()
        )

        peak_day_pm25 = float(
            day_df["pm25"].max()
        )

        peak_day_aqi = int(
            day_df["aqi"].max()
        )

        category = str(
            day_df["aqi_category"]
            .mode()
            .iloc[0]
        )

        st.metric(
            "Peak AQI",
            peak_day_aqi,
            category,
        )

        st.metric(
            "Average PM2.5",
            f"{avg_day_pm25:.1f} µg/m³",
        )

        st.metric(
            "Peak PM2.5",
            f"{peak_day_pm25:.1f} µg/m³",
        )


# ============================================================
# MODEL ROUTING TIMELINE
# ============================================================

st.divider()

st.header("🧠 Forecast Decision Timeline")

st.caption(
    "Validation-selected model for each "
    "forecast horizon."
)

timeline = go.Figure()

for model_name in [
    "Random Forest",
    "LSTM",
]:

    subset = forecast[
        forecast["model_used"] == model_name
    ]

    if subset.empty:
        continue

    timeline.add_trace(
        go.Scatter(
            x=subset["hour_ahead"],
            y=[model_name] * len(subset),
            mode="markers",
            name=model_name,
            marker={
                "size": 12,
            },
            hovertemplate=(
                "H+%{x}"
                "<br>%{y}"
                "<extra></extra>"
            ),
        )
    )

timeline.update_layout(
    height=300,
    template="plotly_dark",
    paper_bgcolor="#0f1b2d",
    plot_bgcolor="#0f1b2d",
    margin={
        "l": 45,
        "r": 25,
        "t": 25,
        "b": 45,
    },
    font={
        "color": "#e2e8f0",
    },
    xaxis={
        "title": "Forecast horizon",
        "dtick": 6,
        "range": [0, 73],
        "showgrid": False,
        "linecolor": "#475569",
    },
    yaxis={
        "title": "Selected model",
        "showgrid": False,
        "linecolor": "#475569",
    },
)

st.plotly_chart(
    timeline,
    width="stretch",
)


# ============================================================
# COMPLETE FORECAST TABLE
# ============================================================

st.divider()

st.header("📋 Complete 72-Hour Forecast")

table_df = forecast[
    [
        "timestamp",
        "hour_ahead",
        "day",
        "model_used",
        "pm25",
        "pm25_24h_average",
        "aqi",
        "aqi_category",
    ]
].copy()

table_df["timestamp"] = (
    table_df["timestamp"]
    .dt.strftime("%Y-%m-%d %H:%M")
)

st.dataframe(
    table_df,
    width="stretch",
    hide_index=True,
    height=540,
    column_config={
        "timestamp": "Forecast time",
        "hour_ahead": "Horizon",
        "day": "Day",
        "model_used": "Model",
        "pm25": st.column_config.NumberColumn(
            "PM2.5",
            format="%.2f",
        ),
        "pm25_24h_average": st.column_config.NumberColumn(
            "24h PM2.5",
            format="%.2f",
        ),
        "aqi": st.column_config.NumberColumn(
            "AQI",
            format="%d",
        ),
        "aqi_category": "Category",
    },
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AirSense-AI • Latest available forecast • "
    f"Source observation: "
    f"{latest_observation.strftime('%Y-%m-%d %H:%M UTC')} • "
    f"Forecast ends: "
    f"{latest_forecast.strftime('%Y-%m-%d %H:%M UTC')}"
)

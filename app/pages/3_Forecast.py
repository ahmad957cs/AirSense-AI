from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.prediction.forecast_engine import ForecastEngine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AirSense-AI | Forecast",
    page_icon="🌫️",
    layout="wide",
)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(14,165,233,.08),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(99,102,241,.07),
                transparent 25%
            ),
            #07111f;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #07111f;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PRODUCTION FORECAST
# ============================================================

@st.cache_data(ttl=300)
def load_production_forecast() -> pd.DataFrame:
    """
    Production forecasting path.

    Features are loaded through Hopsworks Feature View by
    ForecastEngine. No local CSV is used as the production
    inference source.
    """
    engine = ForecastEngine()

    forecast = engine.forecast()

    if forecast is None or forecast.empty:
        raise RuntimeError(
            "ForecastEngine returned no forecast data."
        )

    forecast = forecast.copy()

    if "timestamp" not in forecast.columns:
        raise RuntimeError(
            "ForecastEngine output is missing 'timestamp'."
        )

    forecast["timestamp"] = pd.to_datetime(
        forecast["timestamp"],
        utc=True,
        errors="coerce",
    )

    forecast = (
        forecast.dropna(subset=["timestamp"])
        .sort_values("hour_ahead")
        .reset_index(drop=True)
    )

    if len(forecast) != 72:
        raise RuntimeError(
            f"Expected 72 forecast hours, "
            f"but received {len(forecast)}."
        )

    return forecast


# ============================================================
# LOAD PRODUCTION DATA
# ============================================================

try:
    forecast = load_production_forecast()

except Exception as exc:
    st.title("🌫️ 72-Hour Air Quality Outlook")
    st.subheader("AirSense-AI Predictive Intelligence")

    st.error(
        "A production forecast is currently unavailable."
    )

    st.warning(
        str(exc)
    )

    st.info(
        "The dashboard requires a valid continuous 72-hour "
        "window from the Hopsworks Feature Store. "
        "No local CSV fallback is used for production inference."
    )

    st.stop()


# ============================================================
# SUMMARY VALUES
# ============================================================

latest_observation = (
    forecast["timestamp"].min()
    - pd.Timedelta(hours=1)
)

latest_forecast = forecast["timestamp"].max()

current_pm25 = float(
    forecast["pm25"].iloc[0]
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
    pd.to_datetime(
        latest_observation,
        utc=True,
    ).to_pydatetime()
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

    st.subheader("Forecast")

    st.write(
        "Horizon: **72 hours**"
    )

    st.write(
        "Resolution: **Hourly**"
    )

    st.write(
        "Models: **LSTM + Random Forest**"
    )

    st.divider()

    st.caption(
        "Latest available forecast"
    )

    st.caption(
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


if age_hours <= 2:

    st.success(
        "● Fresh feature data"
    )

elif age_hours <= 24:

    st.warning(
        "● Feature data is becoming stale"
    )

else:

    st.warning(
        "● Historical / stale feature data"
    )

st.caption(
    f"Latest source observation: "
    f"{latest_observation.strftime('%Y-%m-%d %H:%M UTC')} "
    f"• Data age: {age_hours:.1f} hours"
)


# ============================================================
# FORECAST SNAPSHOT
# ============================================================

st.divider()

st.header(
    "Forecast Snapshot"
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Current PM2.5",
        f"{current_pm25:.1f}",
        "µg/m³",
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
        f"{average_pm25:.1f}",
        "µg/m³",
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

st.header(
    "🤖 Horizon-wise Model Selection"
)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "LSTM Horizons",
        int(model_counts.get("LSTM", 0)),
        "selected by validation",
    )

with m2:
    st.metric(
        "Random Forest Horizons",
        int(
            model_counts.get(
                "Random Forest",
                0,
            )
        ),
        "selected by validation",
    )

with m3:
    st.metric(
        "Forecast Horizon",
        "72h",
        "3 days",
    )


# ============================================================
# PM2.5 CHART
# ============================================================

st.divider()

st.header(
    "📈 PM2.5 Forecast Trajectory"
)

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
            "size": 5,
        },
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.06)",
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
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(7,17,31,0.45)",
    margin={
        "l": 20,
        "r": 20,
        "t": 30,
        "b": 30,
    },
    hovermode="x unified",
    xaxis={
        "title": "Forecast time",
        "showgrid": False,
    },
    yaxis={
        "title": "PM2.5 (µg/m³)",
        "gridcolor": "rgba(148,163,184,0.08)",
    },
)

st.plotly_chart(
    fig_pm25,
    width="stretch",
)


# ============================================================
# AQI CHART
# ============================================================

st.header(
    "🟢 AQI Outlook"
)

fig_aqi = go.Figure()

fig_aqi.add_hrect(
    y0=0,
    y1=50,
    fillcolor="rgba(34,197,94,0.05)",
    line_width=0,
)

fig_aqi.add_hrect(
    y0=50,
    y1=100,
    fillcolor="rgba(234,179,8,0.06)",
    line_width=0,
)

fig_aqi.add_hrect(
    y0=100,
    y1=150,
    fillcolor="rgba(249,115,22,0.07)",
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
            "size": 5,
        },
        hovertemplate=(
            "<b>%{x|%b %d %H:%M}</b><br>"
            "AQI: %{y}"
            "<extra></extra>"
        ),
    )
)

fig_aqi.update_layout(
    height=390,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(7,17,31,0.45)",
    margin={
        "l": 20,
        "r": 20,
        "t": 30,
        "b": 30,
    },
    hovermode="x unified",
    xaxis={
        "title": "Forecast time",
        "showgrid": False,
    },
    yaxis={
        "title": "AQI",
        "gridcolor": "rgba(148,163,184,0.08)",
    },
)

st.plotly_chart(
    fig_aqi,
    width="stretch",
)


# ============================================================
# THREE DAY SUMMARY
# ============================================================

st.divider()

st.header(
    "📅 Three-Day Outlook"
)

d1, d2, d3 = st.columns(3)

day_columns = {
    "Day 1": d1,
    "Day 2": d2,
    "Day 3": d3,
}

for day_name, column in day_columns.items():

    day_df = forecast[
        forecast["day"] == day_name
    ]

    if day_df.empty:
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

    category = (
        day_df["aqi_category"]
        .mode()
        .iloc[0]
    )

    with column:

        st.subheader(
            day_name
        )

        st.metric(
            "Peak AQI",
            peak_day_aqi,
            category,
        )

        st.metric(
            "Average PM2.5",
            f"{avg_day_pm25:.1f}",
            "µg/m³",
        )

        st.metric(
            "Peak PM2.5",
            f"{peak_day_pm25:.1f}",
            "µg/m³",
        )


# ============================================================
# MODEL ROUTING
# ============================================================

st.divider()

st.header(
    "🧠 Forecast Decision Timeline"
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
                "size": 11,
            },
            hovertemplate=(
                "H+%{x}"
                "<br>%{y}"
                "<extra></extra>"
            ),
        )
    )

timeline.update_layout(
    height=280,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(7,17,31,0.45)",
    margin={
        "l": 20,
        "r": 20,
        "t": 20,
        "b": 20,
    },
    xaxis={
        "title": "Forecast horizon",
        "dtick": 6,
        "range": [0, 73],
        "showgrid": False,
    },
    yaxis={
        "title": "Selected model",
        "showgrid": False,
    },
)

st.plotly_chart(
    timeline,
    width="stretch",
)


# ============================================================
# FULL FORECAST TABLE
# ============================================================

st.divider()

st.header(
    "📋 Complete 72-Hour Forecast"
)

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
    "AirSense-AI • Latest available production forecast • "
    f"Source observation: "
    f"{latest_observation.strftime('%Y-%m-%d %H:%M UTC')} • "
    f"Forecast ends: "
    f"{latest_forecast.strftime('%Y-%m-%d %H:%M UTC')}"
)
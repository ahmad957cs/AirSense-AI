from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = PROJECT_ROOT / "artifacts" / "eda"
PLOTS_DIR = EDA_DIR / "plots"


st.set_page_config(
    page_title="AirSense-AI | EDA",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Exploratory Data Analysis")
st.caption(
    "Historical PM2.5 trends, distributions, feature relationships, and data quality."
)


# ============================================================
# Dataset summary
# ============================================================

summary_path = EDA_DIR / "eda_summary.json"

if summary_path.exists():
    summary = pd.read_json(
        summary_path,
        typ="series",
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        int(summary["rows"]),
    )

    c2.metric(
        "Columns",
        int(summary["columns"]),
    )

    c3.metric(
        "Missing cells",
        int(summary["missing_cells"]),
    )

    c4.metric(
        "Duplicate timestamps",
        int(summary["duplicate_timestamps"]),
    )


# ============================================================
# PM2.5 trend
# ============================================================

st.subheader("PM2.5 Over Time")

plot = PLOTS_DIR / "pm25_time_series.png"

if plot.exists():
    st.image(
        str(plot),
        width="stretch",
    )


# ============================================================
# Distribution
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("PM2.5 Distribution")

    plot = PLOTS_DIR / "pm25_distribution.png"

    if plot.exists():
        st.image(
            str(plot),
            width="stretch",
        )

with col2:
    st.subheader("PM2.5 Boxplot")

    plot = PLOTS_DIR / "pm25_boxplot.png"

    if plot.exists():
        st.image(
            str(plot),
            width="stretch",
        )


# ============================================================
# Temporal patterns
# ============================================================

st.subheader("Temporal Patterns")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Hourly Pattern")

    plot = PLOTS_DIR / "hourly_pm25_pattern.png"

    if plot.exists():
        st.image(
            str(plot),
            width="stretch",
        )

with col2:
    st.markdown("#### Day of Week")

    plot = PLOTS_DIR / "day_of_week_pm25.png"

    if plot.exists():
        st.image(
            str(plot),
            width="stretch",
        )


col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Monthly Pattern")

    plot = PLOTS_DIR / "monthly_pm25_pattern.png"

    if plot.exists():
        st.image(
            str(plot),
            width="stretch",
        )

with col2:
    st.markdown("#### Weekday vs Weekend")

    plot = PLOTS_DIR / "weekend_pm25.png"

    if plot.exists():
        st.image(
            str(plot),
            width="stretch",
        )


# ============================================================
# Rolling statistics
# ============================================================

st.subheader("Rolling Statistics")

plot = PLOTS_DIR / "pm25_rolling_24h.png"

if plot.exists():
    st.image(
        str(plot),
        width="stretch",
    )


# ============================================================
# Correlations
# ============================================================

st.subheader("Feature Correlations")

plot = PLOTS_DIR / "pm25_feature_correlations.png"

if plot.exists():
    st.image(
        str(plot),
        width="stretch",
    )


# ============================================================
# Missing values
# ============================================================

st.subheader("Missing Values")

plot = PLOTS_DIR / "missing_values.png"

if plot.exists():
    st.image(
        str(plot),
        width="stretch",
    )

missing_path = EDA_DIR / "missing_values.csv"

if missing_path.exists():
    missing_df = pd.read_csv(
        missing_path
    )

    st.dataframe(
        missing_df,
        width="stretch",
        hide_index=True,
    )


# ============================================================
# Timestamp gaps
# ============================================================

st.subheader("Timestamp Continuity")

gap_path = (
    EDA_DIR
    / "timestamp_gap_analysis.csv"
)

if gap_path.exists():

    gap_df = pd.read_csv(
        gap_path
    )

    st.dataframe(
        gap_df,
        width="stretch",
        hide_index=True,
    )

gap_summary_path = (
    EDA_DIR
    / "timestamp_gap_summary.json"
)

if gap_summary_path.exists():

    gap_summary = pd.read_json(
        gap_summary_path,
        typ="series",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "1-hour intervals",
        int(
            gap_summary["one_hour_intervals"]
        ),
    )

    c2.metric(
        "Non-hourly intervals",
        int(
            gap_summary["non_hourly_intervals"]
        ),
    )

    c3.metric(
        "Largest gap",
        str(
            gap_summary["largest_gap"]
        ),
    )

st.success(
    "EDA artifacts loaded successfully."
)

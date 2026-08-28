from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
)


RF_METRICS_PATH = (
    METRICS_DIR
    / "random_forest_validation_metrics.csv"
)

XGB_METRICS_PATH = (
    METRICS_DIR
    / "xgboost_validation_metrics.csv"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AirSense-AI | Model Performance",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

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

    [data-testid="stSidebar"] {
        background: #101a2b;
        border-right: 1px solid #263852;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    p {
        color: #cbd5e1;
    }

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

    [data-testid="stPlotlyChart"] {
        background: #0f1b2d;
        border: 1px solid #2b3f5c;
        border-radius: 16px;
        padding: 6px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #2b3f5c;
        border-radius: 14px;
        overflow: hidden;
    }

    hr {
        border-color: #263852 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📊 AirSense-AI")

    st.caption(
        "Model evaluation and validation metrics"
    )

    st.divider()

    st.write("Evaluation")
    st.write("• MAE")
    st.write("• RMSE")
    st.write("• R²")

    st.divider()

    st.caption(
        "Models evaluated"
    )

    st.write("Random Forest")
    st.write("XGBoost")


# ============================================================
# LOAD METRICS
# ============================================================

@st.cache_data(ttl=300)
def load_metrics(path: str) -> pd.DataFrame:

    file_path = Path(path)

    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    return df


rf_metrics = load_metrics(
    str(RF_METRICS_PATH)
)

xgb_metrics = load_metrics(
    str(XGB_METRICS_PATH)
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "📊 Model Performance"
)

st.subheader(
    "AirSense-AI Forecasting Model Evaluation"
)

st.write(
    "Validation performance across the 72-hour "
    "PM2.5 forecasting horizon."
)


# ============================================================
# DATA AVAILABILITY
# ============================================================

available_models = []

if not rf_metrics.empty:
    available_models.append("Random Forest")

if not xgb_metrics.empty:
    available_models.append("XGBoost")


if not available_models:
    st.warning(
        "No model metric artifacts are currently available."
    )

    st.info(
        "Expected metric files are located in "
        "`artifacts/metrics/`."
    )

    st.stop()


st.success(
    "Metric artifacts loaded successfully."
)


# ============================================================
# METRIC SUMMARY
# ============================================================

st.divider()

st.header(
    "Overall Validation Summary"
)

summary_rows = []


def add_summary_row(
    model_name: str,
    metrics_df: pd.DataFrame,
) -> None:

    required = [
        "MAE",
        "RMSE",
        "R2",
    ]

    if not all(
        column in metrics_df.columns
        for column in required
    ):
        return

    summary_rows.append(
        {
            "Model": model_name,
            "Mean MAE": float(
                metrics_df["MAE"].mean()
            ),
            "Mean RMSE": float(
                metrics_df["RMSE"].mean()
            ),
            "Mean R²": float(
                metrics_df["R2"].mean()
            ),
        }
    )


if not rf_metrics.empty:
    add_summary_row(
        "Random Forest",
        rf_metrics,
    )

if not xgb_metrics.empty:
    add_summary_row(
        "XGBoost",
        xgb_metrics,
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SUMMARY CARDS
# ============================================================

if not summary_df.empty:

    best_mae_index = (
        summary_df["Mean MAE"]
        .idxmin()
    )

    best_rmse_index = (
        summary_df["Mean RMSE"]
        .idxmin()
    )

    best_r2_index = (
        summary_df["Mean R²"]
        .idxmax()
    )

    best_mae_model = summary_df.loc[
        best_mae_index,
        "Model",
    ]

    best_rmse_model = summary_df.loc[
        best_rmse_index,
        "Model",
    ]

    best_r2_model = summary_df.loc[
        best_r2_index,
        "Model",
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Models Evaluated",
            len(summary_df),
        )

    with c2:
        st.metric(
            "Best Mean MAE",
            f"{summary_df.loc[best_mae_index, 'Mean MAE']:.3f}",
            best_mae_model,
        )

    with c3:
        st.metric(
            "Best Mean RMSE",
            f"{summary_df.loc[best_rmse_index, 'Mean RMSE']:.3f}",
            best_rmse_model,
        )

    with c4:
        st.metric(
            "Best Mean R²",
            f"{summary_df.loc[best_r2_index, 'Mean R²']:.3f}",
            best_r2_model,
        )


# ============================================================
# SUMMARY TABLE
# ============================================================

st.divider()

st.subheader(
    "Model Comparison"
)

display_summary = summary_df.copy()

display_summary["Mean MAE"] = (
    display_summary["Mean MAE"]
    .round(4)
)

display_summary["Mean RMSE"] = (
    display_summary["Mean RMSE"]
    .round(4)
)

display_summary["Mean R²"] = (
    display_summary["Mean R²"]
    .round(4)
)

st.dataframe(
    display_summary,
    width="stretch",
    hide_index=True,
)


# ============================================================
# MODEL SELECTOR
# ============================================================

st.divider()

st.header(
    "Horizon-wise Validation Performance"
)

selected_model = st.selectbox(
    "Select model",
    available_models,
)


if selected_model == "Random Forest":
    selected_metrics = rf_metrics
else:
    selected_metrics = xgb_metrics


# ============================================================
# VALIDATE METRIC SCHEMA
# ============================================================

required_columns = [
    "MAE",
    "RMSE",
    "R2",
]

missing_columns = [
    column
    for column in required_columns
    if column not in selected_metrics.columns
]

if missing_columns:

    st.error(
        "Metric artifact is missing required columns:"
    )

    st.code(
        "\n".join(missing_columns)
    )

    st.stop()


# ============================================================
# HORIZON COLUMN
# ============================================================

if "horizon" in selected_metrics.columns:

    horizon_column = "horizon"

elif "hour_ahead" in selected_metrics.columns:

    horizon_column = "hour_ahead"

elif "hour" in selected_metrics.columns:

    horizon_column = "hour"

else:

    selected_metrics = (
        selected_metrics
        .copy()
    )

    selected_metrics[
        "horizon"
    ] = range(
        1,
        len(selected_metrics) + 1,
    )

    horizon_column = "horizon"


plot_df = (
    selected_metrics
    .copy()
    .sort_values(horizon_column)
    .reset_index(drop=True)
)


# ============================================================
# PERFORMANCE CARDS
# ============================================================

mean_mae = float(
    plot_df["MAE"].mean()
)

mean_rmse = float(
    plot_df["RMSE"].mean()
)

mean_r2 = float(
    plot_df["R2"].mean()
)

p1, p2, p3 = st.columns(3)

with p1:
    st.metric(
        "Mean MAE",
        f"{mean_mae:.3f}",
    )

with p2:
    st.metric(
        "Mean RMSE",
        f"{mean_rmse:.3f}",
    )

with p3:
    st.metric(
        "Mean R²",
        f"{mean_r2:.3f}",
    )


# ============================================================
# MAE CHART
# ============================================================

st.divider()

st.subheader(
    f"{selected_model} — MAE by Forecast Horizon"
)

fig_mae = go.Figure()

fig_mae.add_trace(
    go.Scatter(
        x=plot_df[horizon_column],
        y=plot_df["MAE"],
        mode="lines+markers",
        name="MAE",
        line={
            "color": "#38bdf8",
            "width": 3,
        },
        marker={
            "size": 6,
            "color": "#7dd3fc",
        },
        hovertemplate=(
            "Horizon: H+%{x}<br>"
            "MAE: %{y:.4f}"
            "<extra></extra>"
        ),
    )
)

fig_mae.update_layout(
    height=390,
    template="plotly_dark",
    paper_bgcolor="#0f1b2d",
    plot_bgcolor="#0f1b2d",
    font={
        "color": "#e2e8f0",
    },
    margin={
        "l": 45,
        "r": 25,
        "t": 25,
        "b": 45,
    },
    xaxis={
        "title": "Forecast horizon",
        "linecolor": "#475569",
        "showgrid": False,
    },
    yaxis={
        "title": "MAE",
        "gridcolor": "#263852",
        "linecolor": "#475569",
        "zeroline": False,
    },
)

st.plotly_chart(
    fig_mae,
    width="stretch",
)


# ============================================================
# RMSE CHART
# ============================================================

st.subheader(
    f"{selected_model} — RMSE by Forecast Horizon"
)

fig_rmse = go.Figure()

fig_rmse.add_trace(
    go.Scatter(
        x=plot_df[horizon_column],
        y=plot_df["RMSE"],
        mode="lines+markers",
        name="RMSE",
        line={
            "color": "#a78bfa",
            "width": 3,
        },
        marker={
            "size": 6,
            "color": "#c4b5fd",
        },
        hovertemplate=(
            "Horizon: H+%{x}<br>"
            "RMSE: %{y:.4f}"
            "<extra></extra>"
        ),
    )
)

fig_rmse.update_layout(
    height=390,
    template="plotly_dark",
    paper_bgcolor="#0f1b2d",
    plot_bgcolor="#0f1b2d",
    font={
        "color": "#e2e8f0",
    },
    margin={
        "l": 45,
        "r": 25,
        "t": 25,
        "b": 45,
    },
    xaxis={
        "title": "Forecast horizon",
        "linecolor": "#475569",
        "showgrid": False,
    },
    yaxis={
        "title": "RMSE",
        "gridcolor": "#263852",
        "linecolor": "#475569",
        "zeroline": False,
    },
)

st.plotly_chart(
    fig_rmse,
    width="stretch",
)


# ============================================================
# R² CHART
# ============================================================

st.subheader(
    f"{selected_model} — R² by Forecast Horizon"
)

fig_r2 = go.Figure()

fig_r2.add_trace(
    go.Scatter(
        x=plot_df[horizon_column],
        y=plot_df["R2"],
        mode="lines+markers",
        name="R²",
        line={
            "color": "#34d399",
            "width": 3,
        },
        marker={
            "size": 6,
            "color": "#6ee7b7",
        },
        hovertemplate=(
            "Horizon: H+%{x}<br>"
            "R²: %{y:.4f}"
            "<extra></extra>"
        ),
    )
)

fig_r2.update_layout(
    height=390,
    template="plotly_dark",
    paper_bgcolor="#0f1b2d",
    plot_bgcolor="#0f1b2d",
    font={
        "color": "#e2e8f0",
    },
    margin={
        "l": 45,
        "r": 25,
        "t": 25,
        "b": 45,
    },
    xaxis={
        "title": "Forecast horizon",
        "linecolor": "#475569",
        "showgrid": False,
    },
    yaxis={
        "title": "R²",
        "gridcolor": "#263852",
        "linecolor": "#475569",
        "zeroline": False,
    },
)

st.plotly_chart(
    fig_r2,
    width="stretch",
)


# ============================================================
# RAW METRICS TABLE
# ============================================================

st.divider()

st.subheader(
    f"{selected_model} — Complete Validation Metrics"
)

metrics_table = plot_df[
    [
        horizon_column,
        "MAE",
        "RMSE",
        "R2",
    ]
].copy()

metrics_table.columns = [
    "Horizon",
    "MAE",
    "RMSE",
    "R²",
]

metrics_table["Horizon"] = (
    "H+"
    + metrics_table["Horizon"]
    .astype(int)
    .astype(str)
)

metrics_table["MAE"] = (
    metrics_table["MAE"].round(4)
)

metrics_table["RMSE"] = (
    metrics_table["RMSE"].round(4)
)

metrics_table["R²"] = (
    metrics_table["R²"].round(4)
)

st.dataframe(
    metrics_table,
    width="stretch",
    hide_index=True,
    height=520,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AirSense-AI • Model evaluation dashboard • "
    "Metrics are loaded from the training pipeline artifacts."
)

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
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

SHAP_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "explainability"
    / "xgboost_shap_importance.csv"
)

RF_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest"
    / "random_forest_72h.joblib"
)

XGB_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
    / "xgboost_72h.joblib"
)

IMPUTER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessing"
    / "tabular_imputer.joblib"
)

RF_METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "random_forest_validation_metrics.csv"
)

XGB_METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "xgboost_validation_metrics.csv"
)

FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pm25_features.csv"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AirSense-AI | MLOps",
    page_icon="⚙️",
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
# HELPERS
# ============================================================

def file_status(path: Path) -> str:
    return "Available" if path.exists() else "Missing"


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0

    return path.stat().st_size / (1024 * 1024)


def load_metric_summary(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)

        required = ["MAE", "RMSE", "R2"]

        if not all(column in df.columns for column in required):
            return None

        return {
            "MAE": float(df["MAE"].mean()),
            "RMSE": float(df["RMSE"].mean()),
            "R2": float(df["R2"].mean()),
        }

    except Exception:
        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ AirSense-AI")

    st.caption(
        "Monitoring and operational status"
    )

    st.divider()

    st.write("Feature Store")
    st.write("Hopsworks")

    st.write("Automation")
    st.write("GitHub Actions")

    st.write("Inference")
    st.write("72-hour forecasting")

    st.divider()

    st.caption(
        "This page reports local artifact and pipeline "
        "status. It does not modify production resources."
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚙️ MLOps & System Monitoring"
)

st.subheader(
    "AirSense-AI Operational Overview"
)

st.write(
    "Monitoring of model artifacts, forecast outputs, "
    "explainability artifacts, evaluation results, "
    "and production pipeline configuration."
)


# ============================================================
# CORE STATUS
# ============================================================

st.divider()

st.header("System Status")

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric(
        "Forecast Artifact",
        "READY" if FORECAST_PATH.exists() else "MISSING",
    )

with s2:
    st.metric(
        "Random Forest",
        "READY" if RF_MODEL_PATH.exists() else "MISSING",
    )

with s3:
    st.metric(
        "XGBoost",
        "READY" if XGB_MODEL_PATH.exists() else "MISSING",
    )

with s4:
    st.metric(
        "SHAP Artifact",
        "READY" if SHAP_PATH.exists() else "MISSING",
    )


# ============================================================
# FORECAST OUTPUT STATUS
# ============================================================

st.divider()

st.header("Forecast Pipeline")

if FORECAST_PATH.exists():

    try:

        forecast_df = pd.read_csv(
            FORECAST_PATH,
            parse_dates=["timestamp"],
        )

        forecast_rows = len(forecast_df)

        latest_forecast = (
            forecast_df["timestamp"].max()
        )

        peak_aqi = int(
            forecast_df["aqi"].max()
        )

        model_count = (
            forecast_df["model_used"]
            .value_counts()
            .to_dict()
        )

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric(
                "Forecast Rows",
                forecast_rows,
            )

        with p2:
            st.metric(
                "Expected Horizons",
                "72",
            )

        with p3:
            st.metric(
                "Peak AQI",
                peak_aqi,
            )

        with p4:
            st.metric(
                "Latest Forecast",
                latest_forecast.strftime(
                    "%Y-%m-%d %H:%M"
                ),
            )

        if forecast_rows == 72:

            st.success(
                "72-hour forecast artifact is structurally valid."
            )

        else:

            st.warning(
                f"Forecast contains {forecast_rows} rows "
                "instead of the expected 72."
            )

        routing_df = pd.DataFrame(
            {
                "Model": list(model_count.keys()),
                "Selected horizons": list(
                    model_count.values()
                ),
            }
        )

        st.subheader(
            "Forecast Model Routing"
        )

        st.dataframe(
            routing_df,
            width="stretch",
            hide_index=True,
        )

    except Exception as exc:

        st.error(
            f"Unable to read forecast artifact: {exc}"
        )

else:

    st.warning(
        "Forecast artifact is not available."
    )


# ============================================================
# MODEL ARTIFACTS
# ============================================================

st.divider()

st.header("Model Artifacts")

model_status_df = pd.DataFrame(
    [
        {
            "Model": "Random Forest",
            "Status": file_status(RF_MODEL_PATH),
            "Size (MB)": round(
                file_size_mb(RF_MODEL_PATH),
                2,
            ),
        },
        {
            "Model": "XGBoost",
            "Status": file_status(XGB_MODEL_PATH),
            "Size (MB)": round(
                file_size_mb(XGB_MODEL_PATH),
                2,
            ),
        },
        {
            "Model": "Tabular Imputer",
            "Status": file_status(IMPUTER_PATH),
            "Size (MB)": round(
                file_size_mb(IMPUTER_PATH),
                4,
            ),
        },
    ]
)

st.dataframe(
    model_status_df,
    width="stretch",
    hide_index=True,
)


# ============================================================
# EVALUATION ARTIFACTS
# ============================================================

st.divider()

st.header("Model Evaluation")

rf_summary = load_metric_summary(
    RF_METRICS_PATH
)

xgb_summary = load_metric_summary(
    XGB_METRICS_PATH
)

evaluation_rows = []

if rf_summary is not None:

    evaluation_rows.append(
        {
            "Model": "Random Forest",
            "Mean MAE": round(
                rf_summary["MAE"],
                4,
            ),
            "Mean RMSE": round(
                rf_summary["RMSE"],
                4,
            ),
            "Mean R²": round(
                rf_summary["R2"],
                4,
            ),
            "Status": "Available",
        }
    )

if xgb_summary is not None:

    evaluation_rows.append(
        {
            "Model": "XGBoost",
            "Mean MAE": round(
                xgb_summary["MAE"],
                4,
            ),
            "Mean RMSE": round(
                xgb_summary["RMSE"],
                4,
            ),
            "Mean R²": round(
                xgb_summary["R2"],
                4,
            ),
            "Status": "Available",
        }
    )

if evaluation_rows:

    st.dataframe(
        pd.DataFrame(evaluation_rows),
        width="stretch",
        hide_index=True,
    )

else:

    st.info(
        "No evaluation metric artifacts are currently available."
    )


# ============================================================
# EXPLAINABILITY STATUS
# ============================================================

st.divider()

st.header("Explainability")

if SHAP_PATH.exists():

    try:

        shap_df = pd.read_csv(
            SHAP_PATH
        )

        if {
            "feature",
            "mean_abs_shap",
        }.issubset(shap_df.columns):

            shap_df = (
                shap_df
                .sort_values(
                    "mean_abs_shap",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            top_feature = (
                shap_df.iloc[0]["feature"]
            )

            top_value = float(
                shap_df.iloc[0]["mean_abs_shap"]
            )

            e1, e2, e3 = st.columns(3)

            with e1:
                st.metric(
                    "Features Explained",
                    len(shap_df),
                )

            with e2:
                st.metric(
                    "Top Feature",
                    str(top_feature),
                )

            with e3:
                st.metric(
                    "Top Mean |SHAP|",
                    f"{top_value:.3f}",
                )

            st.success(
                "SHAP explainability artifact is available."
            )

        else:

            st.warning(
                "SHAP artifact exists but has an unexpected schema."
            )

    except Exception as exc:

        st.warning(
            f"Unable to read SHAP artifact: {exc}"
        )

else:

    st.warning(
        "SHAP explainability artifact is not available."
    )


# ============================================================
# FEATURE DATA STATUS
# ============================================================

st.divider()

st.header("Feature Data")

if FEATURE_DATA_PATH.exists():

    try:

        feature_df = pd.read_csv(
            FEATURE_DATA_PATH,
            parse_dates=["timestamp"],
        )

        f1, f2, f3 = st.columns(3)

        with f1:
            st.metric(
                "Feature Rows",
                len(feature_df),
            )

        with f2:
            st.metric(
                "Feature Columns",
                len(feature_df.columns),
            )

        with f3:
            if "timestamp" in feature_df.columns:

                latest_feature = (
                    feature_df["timestamp"].max()
                )

                st.metric(
                    "Latest Feature",
                    latest_feature.strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                )

            else:

                st.metric(
                    "Timestamp",
                    "Missing",
                )

        st.success(
            "Local feature dataset is available."
        )

    except Exception as exc:

        st.error(
            f"Unable to read feature dataset: {exc}"
        )

else:

    st.warning(
        "Local feature dataset is not available."
    )


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

st.divider()

st.header("Production Pipeline Configuration")

configuration_df = pd.DataFrame(
    [
        {
            "Component": "Data source",
            "Technology": "OpenAQ",
            "Role": "Hourly PM2.5 observations",
        },
        {
            "Component": "Feature Store",
            "Technology": "Hopsworks",
            "Role": "Feature storage and retrieval",
        },
        {
            "Component": "Automation",
            "Technology": "GitHub Actions",
            "Role": "Hourly feature pipeline",
        },
        {
            "Component": "Forecast horizon",
            "Technology": "72 hours",
            "Role": "Hourly PM2.5 prediction",
        },
        {
            "Component": "Forecast models",
            "Technology": "Random Forest + LSTM",
            "Role": "Horizon-wise model routing",
        },
        {
            "Component": "Explainability",
            "Technology": "SHAP",
            "Role": "XGBoost feature importance",
        },
        {
            "Component": "API",
            "Technology": "FastAPI",
            "Role": "Inference service",
        },
        {
            "Component": "Dashboard",
            "Technology": "Streamlit",
            "Role": "Monitoring and visualization",
        },
    ]
)

st.dataframe(
    configuration_df,
    width="stretch",
    hide_index=True,
)


# ============================================================
# LAST CHECK
# ============================================================

st.divider()

st.header("Operational Checks")

checks = [
    (
        "72-hour forecast artifact",
        FORECAST_PATH.exists(),
    ),
    (
        "Random Forest artifact",
        RF_MODEL_PATH.exists(),
    ),
    (
        "XGBoost artifact",
        XGB_MODEL_PATH.exists(),
    ),
    (
        "Tabular preprocessing artifact",
        IMPUTER_PATH.exists(),
    ),
    (
        "SHAP artifact",
        SHAP_PATH.exists(),
    ),
    (
        "Random Forest metrics",
        RF_METRICS_PATH.exists(),
    ),
    (
        "XGBoost metrics",
        XGB_METRICS_PATH.exists(),
    ),
    (
        "Feature dataset",
        FEATURE_DATA_PATH.exists(),
    ),
]


for check_name, passed in checks:

    if passed:
        st.success(
            f"✓ {check_name}"
        )

    else:
        st.warning(
            f"⚠ {check_name} unavailable"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AirSense-AI • MLOps dashboard • "
    f"Checked at "
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)

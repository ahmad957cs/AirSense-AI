from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAP_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "explainability"
    / "xgboost_shap_importance.csv"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AirSense-AI | Explainability",
    page_icon="🧠",
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

    h1, h2, h3 {
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
    st.title("🧠 AirSense-AI")

    st.caption(
        "Model explainability and feature importance"
    )

    st.divider()

    st.write("Explained model")
    st.write("XGBoost")

    st.divider()

    st.write("Explanation method")
    st.write("SHAP")

    st.divider()

    st.caption(
        "Based on the first forecast horizon."
    )


# ============================================================
# LOAD SHAP RESULTS
# ============================================================

@st.cache_data(ttl=300)
def load_shap_results(path: str) -> pd.DataFrame:

    file_path = Path(path)

    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


shap_df = load_shap_results(
    str(SHAP_PATH)
)


# ============================================================
# VALIDATE
# ============================================================

if shap_df.empty:
    st.error(
        "SHAP explanation artifact was not found."
    )

    st.info(
        "Run:"
    )

    st.code(
        "python -m src.explainability.shap_analysis"
    )

    st.stop()


required_columns = [
    "feature",
    "mean_abs_shap",
]

missing_columns = [
    column
    for column in required_columns
    if column not in shap_df.columns
]

if missing_columns:
    st.error(
        "SHAP artifact is missing required columns."
    )

    st.code(
        "\n".join(missing_columns)
    )

    st.stop()


shap_df = (
    shap_df[
        required_columns
    ]
    .dropna()
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🧠 Model Explainability"
)

st.subheader(
    "XGBoost SHAP Feature Importance"
)

st.write(
    "SHAP identifies which input features have the strongest "
    "influence on the XGBoost PM2.5 forecast."
)


# ============================================================
# SUMMARY
# ============================================================

st.divider()

top_feature = (
    shap_df.iloc[0]["feature"]
)

top_importance = float(
    shap_df.iloc[0]["mean_abs_shap"]
)

feature_count = len(shap_df)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Features Explained",
        feature_count,
    )

with c2:
    st.metric(
        "Most Influential Feature",
        str(top_feature),
    )

with c3:
    st.metric(
        "Top SHAP Importance",
        f"{top_importance:.3f}",
    )


# ============================================================
# TOP 15 FEATURES
# ============================================================

st.divider()

st.header(
    "Top 15 Influential Features"
)

top15 = (
    shap_df
    .head(15)
    .sort_values(
        "mean_abs_shap",
        ascending=True,
    )
)

fig = px.bar(
    top15,
    x="mean_abs_shap",
    y="feature",
    orientation="h",
    title="Mean Absolute SHAP Importance",
    labels={
        "mean_abs_shap": "Mean |SHAP value|",
        "feature": "Feature",
    },
)

fig.update_layout(
    height=650,
    template="plotly_dark",
    paper_bgcolor="#0f1b2d",
    plot_bgcolor="#0f1b2d",
    font={
        "color": "#e2e8f0",
    },
    margin={
        "l": 20,
        "r": 30,
        "t": 60,
        "b": 40,
    },
)

fig.update_xaxes(
    gridcolor="#263852",
    zeroline=False,
)

fig.update_yaxes(
    gridcolor="rgba(0,0,0,0)",
)

st.plotly_chart(
    fig,
    width="stretch",
)


# ============================================================
# INTERPRETATION
# ============================================================

st.divider()

st.header(
    "Interpretation"
)

st.info(
    f"The strongest feature in the current explanation is "
    f"**{top_feature}**, with a mean absolute SHAP importance "
    f"of **{top_importance:.3f}**."
)

st.caption(
    "Mean absolute SHAP values describe feature influence "
    "magnitude. They do not by themselves indicate whether "
    "a feature increases or decreases the prediction."
)


# ============================================================
# COMPLETE TABLE
# ============================================================

st.divider()

st.header(
    "Complete SHAP Feature Ranking"
)

display_df = shap_df.copy()

display_df.insert(
    0,
    "Rank",
    range(
        1,
        len(display_df) + 1,
    ),
)

display_df[
    "mean_abs_shap"
] = display_df[
    "mean_abs_shap"
].round(6)

display_df.columns = [
    "Rank",
    "Feature",
    "Mean |SHAP|",
]

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    height=650,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AirSense-AI • XGBoost explainability • "
    "SHAP importance generated from the current model artifact."
)

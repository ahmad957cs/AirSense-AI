from pathlib import Path
import sys

import streamlit as st


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AirSense-AI | Islamabad",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL THEME
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #07111f;
        }

        [data-testid="stSidebar"] {
            background-color: #06131f;
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }

        h1, h2, h3, h4 {
            color: #f8fafc !important;
        }

        p, li {
            color: #cbd5e1;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌫️ AirSense-AI")
    st.caption("Islamabad Air Quality Intelligence")

    st.divider()

    st.subheader("📍 Monitoring Location")

    st.write("**Islamabad, Pakistan**")
    st.write("F-7 Monitoring Station")
    st.caption("Sensor ID: 13137731")

    st.divider()

    st.subheader("Platform")

    st.write("📈 72-Hour Forecasting")
    st.write("🟢 AQI Intelligence")
    st.write("🔬 SHAP Explainability")
    st.write("⚙️ MLOps Monitoring")


# ============================================================
# HERO
# ============================================================

st.markdown("## 🌫️ AirSense-AI")

st.markdown(
    "# **Islamabad Air Quality Outlook**"
)

st.write(
    "An intelligent 72-hour air-quality platform combining "
    "PM2.5 forecasting, AQI interpretation, health alerts, "
    "explainable AI, and MLOps."
)

st.caption(
    "📍 Islamabad, Pakistan  •  F-7 Station  •  Sensor 13137731  •  "
    "72-hour hourly forecasting"
)

st.divider()


# ============================================================
# SYSTEM STATUS
# ============================================================

st.success(
    "🟢 AirSense-AI platform is ready. "
    "Use the navigation menu to explore the complete system."
)


# ============================================================
# PLATFORM AT A GLANCE
# ============================================================

st.subheader("Platform at a Glance")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        label="Forecast Window",
        value="72h",
        delta="Hourly outlook",
    )

with c2:
    st.metric(
        label="Forecasting Models",
        value="2",
        delta="LSTM + Random Forest",
    )

with c3:
    st.metric(
        label="AQI Categories",
        value="6",
        delta="Health classification",
    )

with c4:
    st.metric(
        label="Explainability",
        value="SHAP",
        delta="Feature insights",
    )


# ============================================================
# INTELLIGENCE LAYERS
# ============================================================

st.divider()

st.subheader("From Sensor Data to Actionable Insight")

f1, f2, f3, f4 = st.columns(4)

with f1:
    with st.container(border=True):
        st.markdown("### 🎯 Forecasting")
        st.write(
            "Generate an hourly 72-hour PM2.5 forecast "
            "using sequence and tree-based models with "
            "horizon-wise model routing."
        )

with f2:
    with st.container(border=True):
        st.markdown("### 🟢 AQI Intelligence")
        st.write(
            "Translate PM2.5 predictions into understandable "
            "AQI categories and health-oriented alerts."
        )

with f3:
    with st.container(border=True):
        st.markdown("### 🔍 Explainable AI")
        st.write(
            "Use SHAP-based feature importance to understand "
            "which variables influence the XGBoost model."
        )

with f4:
    with st.container(border=True):
        st.markdown("### ⚙️ MLOps")
        st.write(
            "Inspect artifacts, evaluation results, feature "
            "data, model status, and pipeline configuration."
        )


# ============================================================
# ARCHITECTURE
# ============================================================

st.divider()

st.subheader("End-to-End Architecture")

st.info(
    "🌐 OpenAQ  →  Feature Engineering  →  Hopsworks  →  "
    "Model Routing  →  PM2.5 Forecast  →  AQI  →  "
    "SHAP  →  Streamlit / FastAPI  →  MLOps"
)

st.write(
    "The platform separates data ingestion, feature engineering, "
    "feature storage, forecasting, intelligence, serving, "
    "and monitoring into modular layers."
)


# ============================================================
# EXPLORE
# ============================================================

st.divider()

st.subheader("Explore the AirSense-AI Workspace")

e1, e2, e3 = st.columns(3)

with e1:
    with st.container(border=True):
        st.markdown("### 📊 Explore the Data")
        st.write(
            "Use the EDA page to inspect historical PM2.5 "
            "patterns, distributions, correlations, rolling "
            "statistics, and data quality."
        )

with e2:
    with st.container(border=True):
        st.markdown("### 🔮 Explore the Forecast")
        st.write(
            "Use the Forecast page to examine the 72-hour "
            "PM2.5 trajectory, AQI outlook, three-day summary, "
            "health alerts, and model routing."
        )

with e3:
    with st.container(border=True):
        st.markdown("### 🧠 Inspect the ML System")
        st.write(
            "Use Model Performance, Explainability, and MLOps "
            "to inspect the system beyond its final predictions."
        )


# ============================================================
# PROJECT HIGHLIGHTS
# ============================================================

st.divider()

st.subheader("Project Highlights")

h1, h2 = st.columns(2)

with h1:
    st.markdown("### 🏗️ Engineering")
    st.write(
        "• Hopsworks Feature Store\n"
        "• GitHub Actions automation\n"
        "• FastAPI inference layer\n"
        "• Streamlit dashboard\n"
        "• Reproducible artifacts"
    )

with h2:
    st.markdown("### 🧪 AI / ML")
    st.write(
        "• LSTM forecasting\n"
        "• Random Forest forecasting\n"
        "• XGBoost evaluation\n"
        "• Horizon-wise model selection\n"
        "• SHAP explainability"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AirSense-AI • Islamabad, Pakistan • "
    "72-Hour PM2.5 Forecasting & Air-Quality Intelligence"
)
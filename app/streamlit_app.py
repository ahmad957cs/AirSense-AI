from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st


st.set_page_config(
    page_title="AirSense-AI",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 AirSense-AI")
st.subheader(
    "3-Day Hourly Air Quality Forecasting Platform"
)

st.markdown(
    """
    ### Welcome to AirSense-AI

    Use the navigation panel to explore:

    - Historical EDA and data quality
    - PM2.5 trends and patterns
    - 72-hour forecasting
    - Model performance
    - MLOps information

    The production forecasting engine will provide the
    real 72-hour PM2.5 and AQI predictions.
    """
)

st.info(
    "Select a page from the sidebar to begin."
)

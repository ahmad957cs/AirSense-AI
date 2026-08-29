# 🌫️ AirSense-AI

### Intelligent 72-Hour PM2.5 Forecasting and Air-Quality Intelligence Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://airsense-aigit-2w7ihp5tfvfq9jpncu8ptw.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.60-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Hopsworks](https://img.shields.io/badge/Hopsworks-Feature%20Store-1E88E5)](https://www.hopsworks.ai/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Automated%20Pipelines-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)

---

## 🚀 Live Demo

### Streamlit Application

👉 **[Open AirSense-AI Live Demo](https://airsense-aigit-2w7ihp5tfvfq9jpncu8ptw.streamlit.app/)**

The deployed application provides an interactive interface for:

- 72-hour PM2.5 forecasting
- AQI estimation and classification
- Horizon-wise model routing
- Model performance evaluation
- SHAP-based explainability
- AQI health alerts
- MLOps and operational monitoring

> **Demo note:** The deployed dashboard uses the latest verified forecast artifact available in the repository. Because the forecast artifact is generated from the data pipeline, it can become stale until a newer verified forecast is produced.

---

## 📄 Project Report

### Complete Project Report

**[📑 View Complete Project Report ](https://docs.google.com/document/d/1g2WSRWkwDVfOJpO8dbFkVZo9s9S7v-wx/edit?usp=sharing&ouid=103094263755262699172&rtpof=true&sd=true)**

The final report will document the complete methodology, data pipeline, feature engineering, model development, evaluation, model selection, system architecture, deployment, limitations, and conclusions.

---

# 📌 Overview

**AirSense-AI** is an end-to-end air-quality intelligence platform designed to forecast **PM2.5 concentrations for the next 72 hours** and translate those predictions into **AQI information and health-oriented alerts**.

The system combines:

- automated air-quality data ingestion
- time-series feature engineering
- Hopsworks Feature Store integration
- machine-learning and deep-learning forecasting
- horizon-wise model selection
- AQI computation
- explainable AI using SHAP
- FastAPI inference services
- Streamlit visualization
- GitHub Actions automation
- MLOps-oriented monitoring

The project goes beyond a standalone machine-learning model by demonstrating a **complete production-style ML workflow from data ingestion to deployment and monitoring**.

---

# 🎯 Problem Statement

Air pollution varies over time and can change significantly throughout the day. PM2.5 is an important indicator for understanding short-term air-quality conditions.

A useful forecasting platform therefore needs to do more than report the current measurement. It should be able to:

1. collect historical and recent sensor observations
2. validate and process time-series data
3. construct forecasting features
4. predict future PM2.5 concentrations
5. select suitable models across forecast horizons
6. convert predictions into AQI information
7. communicate potential health risks
8. expose results through an accessible interface
9. provide reproducible ML and MLOps artifacts

AirSense-AI was developed to address this end-to-end engineering problem.

---

# 🧠 Core Research & Engineering Objective

> **Build an automated, explainable, and deployment-oriented system capable of generating a 72-hour hourly PM2.5 forecast with model selection, AQI interpretation, health alerts, and operational monitoring.**

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │       OpenAQ        │
                         │  Hourly PM2.5 Data  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Ingestion    │
                         │ Validation / Merge  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         │ Lags / Rolling /    │
                         │ Time Features       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Hopsworks Feature   │
                         │       Store         │
                         └──────────┬──────────┘
                                    │
                                    ▼
              ┌────────────────────────────────────────┐
              │           Forecasting Layer             │
              │                                        │
              │   Random Forest ───────┐               │
              │                         ├──► 72-Hour    │
              │   LSTM ────────────────┘    Forecast   │
              └────────────────────┬───────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ Horizon-wise Model  │
                         │      Selection      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    PM2.5 → AQI      │
                         │ Health Classification│
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
             ┌────────────┐  ┌──────────────┐  ┌──────────────┐
             │    SHAP    │  │   FastAPI    │  │  Streamlit   │
             │Explainability│ │ Inference API│  │   Dashboard  │
             └────────────┘  └──────────────┘  └───────┬──────┘
                                                       │
                                                       ▼
                                               ┌─────────────┐
                                               │    MLOps    │
                                               │ Monitoring  │
                                               └─────────────┘
🔒 Locked Architecture

The project follows a modular architecture rather than a single monolithic prediction script.

Data Layer
OpenAQ sensor observations
Historical processing
Data validation
Sensor coverage analysis
Feature Layer
Hourly time features
Cyclical encodings
Lag features
Rolling statistics
Gap-aware time-series processing
Feature Store
Hopsworks Feature Store
Feature Group
Feature View
Historical feature retrieval
Forecasting Layer
Random Forest
LSTM
Horizon-wise model routing
72-hour hourly predictions
Intelligence Layer
PM2.5 → AQI conversion
AQI category classification
Health-oriented alerts
SHAP explainability
Serving Layer
FastAPI
Streamlit
Automation / MLOps
GitHub Actions
Automated hourly feature pipeline
Model and evaluation artifacts
Operational monitoring dashboard
📊 Forecasting Approach

AirSense-AI generates an hourly 72-hour forecast.

Instead of treating all forecast horizons as identical, the system supports horizon-wise model selection, allowing different forecast horizons to use different models according to validation performance.
For the current verified forecast artifact:

| Model         | Selected Horizons |
| ------------- | ----------------: |
| LSTM          |                47 |
| Random Forest |                25 |
| **Total**     |            **72** |

This routing strategy allows the forecasting system to exploit the relative strengths of different models across the prediction horizon.

🤖 Machine Learning Models
Random Forest

Random Forest is used as one of the classical forecasting models.

It provides:

nonlinear regression
tree-based learning
a classical forecasting baseline
horizon-specific predictive evaluation

The trained Random Forest artifact is approximately 5.3 GB and is therefore not stored in this GitHub repository.

XGBoost

XGBoost is included as a classical gradient-boosting model for model evaluation and explainability.

It provides:

nonlinear tabular learning
gradient boosting
strong predictive modeling for engineered features
compatibility with SHAP-based explainability
LSTM

A Long Short-Term Memory network is used as the deep-learning forecasting component.

LSTM is appropriate for this project because PM2.5 forecasting is a sequential time-series problem where previous observations influence future predictions.

The trained LSTM artifact is included in the repository:

models/lstm/airsense_lstm_72h_best.keras
🧪 Feature Engineering

The production feature pipeline constructs 43 forecasting features.

Temporal Features
hour
day of week
day of month
month
day of year
ISO week number
weekend indicator
Cyclical Encodings
hour sine / cosine
day-of-week sine / cosine
Lag Features
1-hour lag
3-hour lag
6-hour lag
12-hour lag
24-hour lag
48-hour lag
72-hour lag
Rolling Statistics

For 3-hour, 6-hour, 12-hour, 24-hour, and 72-hour windows:

rolling mean
rolling standard deviation
rolling minimum
rolling maximum
Gap-Aware Processing

The feature engineering pipeline is designed to recognize temporal gaps and reset lag/rolling calculations across non-hourly discontinuities.

This prevents unrelated observations from being treated as continuous sequential history.

🌍 Data Source

The system uses OpenAQ air-quality data.

Current project configuration:

Property	Value
Sensor ID	13137731
Location	F-7, Islamabad
Primary variable	PM2.5
Forecast resolution	Hourly

Processed historical and engineered data are stored in:

data/processed/
🏪 Hopsworks Feature Store

AirSense-AI integrates Hopsworks Feature Store as the centralized feature-management layer.

Current project configuration:

Feature Group
    pm25_hourly_features
    Version: 3

Feature View
    pm25_forecasting_view
    Version: 1

The Feature Store supports:

historical feature storage
feature retrieval
reproducible forecasting inputs
separation between data engineering and inference
📈 AQI Intelligence

The platform translates PM2.5 predictions into AQI information using the configured PM2.5 AQI breakpoints.

Supported categories:

| AQI Range | Category                       |
| --------: | ------------------------------ |
|      0–50 | Good                           |
|    51–100 | Moderate                       |
|   101–150 | Unhealthy for Sensitive Groups |
|   151–200 | Unhealthy                      |
|   201–300 | Very Unhealthy                 |
|   301–500 | Hazardous                      |

The system also generates health-oriented alerts based on the predicted AQI.

Example
Peak AQI:       92
Category:       Moderate
Alert Level:    Advisory
🧠 Explainable AI — SHAP

AirSense-AI includes a dedicated SHAP explainability pipeline.

Current implementation:

Property	Value
Explained model	XGBoost
Method	SHAP
Forecast horizon	First forecast horizon
Features explained	43

Generated artifact:

artifacts/explainability/xgboost_shap_importance.csv

The Explainability dashboard provides:

feature ranking
mean absolute SHAP importance
top influential features
complete feature ranking table
Current top feature

The current explanation identifies:

minimum

as the most influential feature by mean absolute SHAP importance in the generated explanation artifact.

Interpretation note: Mean absolute SHAP values describe the magnitude of feature influence. They do not, by themselves, indicate whether a feature increases or decreases a prediction.

📊 Model Evaluation

The project evaluates forecasting models using:

MAE — Mean Absolute Error
RMSE — Root Mean Squared Error
R² — Coefficient of Determination

Evaluation artifacts are stored under:

artifacts/metrics/

Available evaluation outputs include:

lstm_72h_metrics.csv
lstm_daily_summary.csv
lstm_validation_metrics.csv

random_forest_test_metrics.csv
random_forest_validation_metrics.csv

xgboost_test_metrics.csv
xgboost_validation_metrics.csv

Model-selection artifacts are stored under:

artifacts/model_selection/

These include:

validation leaderboards
hourly model selection
daily validation summaries
final model-selection strategy
timestamp-aligned model comparisons
📱 Streamlit Dashboard

The deployed Streamlit application is organized into six pages.

1. Overview

High-level project summary and system status.

2. EDA

Historical PM2.5 analysis including:

distributions
time-series patterns
hourly trends
day-of-week patterns
monthly trends
rolling statistics
correlation analysis
missing-data analysis
timestamp-gap analysis
3. Forecast

The main forecasting interface provides:

72-hour PM2.5 trajectory
AQI outlook
forecast snapshot
horizon-wise model selection
three-day outlook
model-routing timeline
complete 72-hour forecast table
health-status interpretation
4. Model Performance

Provides:

model comparison
MAE
RMSE
R²
horizon-wise validation performance
complete validation metrics
5. MLOps

Provides:

forecast artifact status
model artifact status
evaluation artifact status
SHAP artifact status
feature-data status
pipeline configuration
operational checks
6. Explainability

Provides:

SHAP feature importance
top influential features
complete feature ranking
explanation summary
⚙️ FastAPI

AirSense-AI also includes a FastAPI inference service.

Main module:

src/api/main.py

Available endpoints:

GET /health
GET /forecast
GET /docs
GET /openapi.json
Health Check
GET /health

Example:

{
  "status": "ok",
  "service": "AirSense-AI"
}
Interactive API Documentation
http://127.0.0.1:8000/docs
🔄 Automated Data Pipeline

AirSense-AI uses GitHub Actions for automated pipeline execution.

Main workflow:

.github/workflows/hourly_feature_pipeline.yml

The workflow is designed to:

retrieve recent OpenAQ observations
validate required credentials
build forecasting features
connect to Hopsworks
update the Feature Store

The workflow is scheduled to run hourly.

🧩 Project Structure
AirSense-AI/
│
├── .github/
│   └── workflows/
│       ├── daily_training.yml
│       └── hourly_feature_pipeline.yml
│
├── api/
│   └── main.py
│
├── app/
│   ├── components/
│   │   ├── charts.py
│   │   ├── metrics.py
│   │   └── sidebar.py
│   │
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_EDA.py
│   │   ├── 3_Forecast.py
│   │   ├── 4_Model_Performance.py
│   │   ├── 5_MLOps.py
│   │   └── 6_Explainability.py
│   │
│   ├── app.py
│   ├── streamlit_app.py
│   └── theme/
│       └── styles.css
│
├── artifacts/
│   ├── eda/
│   ├── explainability/
│   ├── metrics/
│   ├── model_selection/
│   ├── predictions/
│   └── preprocessing/
│
├── config/
│   ├── __init__.py
│   └── config.py
│
├── data/
│   └── processed/
│
├── models/
│   └── lstm/
│
├── scripts/
│   ├── eda/
│   ├── evaluation/
│   ├── feature_store/
│   ├── pipeline/
│   ├── prediction/
│   ├── registry/
│   └── training/
│
├── src/
│   ├── alerts/
│   ├── aqi/
│   ├── backfill/
│   ├── eda/
│   ├── evaluation/
│   ├── explainability/
│   ├── feature_engineering/
│   ├── ingestion/
│   ├── models/
│   ├── monitoring/
│   ├── pipeline/
│   ├── prediction/
│   ├── preprocessing/
│   ├── registry/
│   ├── training/
│   └── utils/
│
├── requirements.txt
├── requirements-ci.txt
├── requirements-training-ci.txt
├── README.md
└── .gitignore

💻 Local Setup
1. Clone the Repository
git clone https://github.com/ahmad957cs/AirSense-AI.git
cd AirSense-AI
2. Create a Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
🔐 Environment Variables

Create a local .env file.

Never commit .env to GitHub.

Example:

OPENAQ_API_KEY=your_openaq_key

HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_PROJECT=AirSense_AI
HOPSWORKS_API_KEY=your_hopsworks_key

OPENWEATHER_API_KEY=your_openweather_key

The exact variables required depend on the pipeline or service being executed.

▶️ Run the Streamlit Application
streamlit run app/streamlit_app.py

Local URL:

http://localhost:8501
▶️ Run FastAPI
uvicorn src.api.main:app --reload

API:

http://127.0.0.1:8000

Swagger:

http://127.0.0.1:8000/docs
🧪 Useful Development Commands
Run SHAP analysis
python -m src.explainability.shap_analysis
Test the forecast artifact
python scripts/prediction/test_forecast.py
Generate the supervised training dataset
python -m scripts.training.create_training_dataset
Recreate the training dataset
python scripts/training/recreate_training_dataset.py
Run the hourly feature pipeline locally
python -m scripts.pipeline.run_hourly_feature_pipeline
📦 Reproducibility Artifacts

The repository contains generated artifacts so the project can be inspected and evaluated without requiring every pipeline stage to be executed from scratch.

Important directories:

artifacts/eda/
artifacts/explainability/
artifacts/metrics/
artifacts/model_selection/
artifacts/predictions/
artifacts/preprocessing/
data/processed/

These contain:

EDA summaries
analytical plots
validation metrics
model-selection results
forecast outputs
preprocessing metadata
SHAP feature importance
processed historical datasets
🔍 Current Forecast Artifact

The repository contains:

artifacts/predictions/latest_72h_forecast.csv

This artifact represents the verified 72-hour hourly forecast used by the Streamlit Forecast page.

The application validates the expected 72 forecast horizons before displaying the complete forecast.

⚙️ MLOps Lifecycle

AirSense-AI demonstrates a complete ML lifecycle:

Data Collection
       ↓
Validation
       ↓
Feature Engineering
       ↓
Feature Store
       ↓
Training / Evaluation
       ↓
Model Selection
       ↓
Forecast Generation
       ↓
AQI / Health Intelligence
       ↓
Explainability
       ↓
API / Dashboard
       ↓
Monitoring

This separation makes the system easier to inspect, reproduce, and maintain.

🔒 Security

Sensitive credentials are intentionally excluded from the repository.

Ignored files include:

.env
.venv/
.hopsworks/
__pycache__/

Credentials should be supplied using environment variables or platform secret management.

The repository intentionally does not contain the oversized Random Forest model artifact.

⚠️ Known Limitations

AirSense-AI is an academic/engineering prototype demonstrating a production-style ML architecture.

1. Random Forest Artifact Size

The trained Random Forest artifact is approximately 5.3 GB and is therefore not stored in this GitHub repository.

The Random Forest model remains part of the forecasting architecture.

2. Live Feature Store Availability

The FastAPI /forecast endpoint depends on a valid recent Feature Store inference window.

If the required production feature window is unavailable, the API returns an explicit 503 Service Unavailable response instead of fabricating a prediction.

3. Forecast Artifact Freshness

The deployed Streamlit dashboard can present the latest verified forecast artifact committed to the repository.

That artifact can become stale until a newer verified forecast is generated.

4. Local Windows Hopsworks Transport

During development, Hopsworks Arrow Flight connectivity on native Windows encountered certificate-verification issues.

The dashboard therefore supports artifact-based visualization independently of that local transport problem.

These limitations document deployment/runtime constraints encountered during implementation rather than hiding them.

📚 Research & Engineering Value

AirSense-AI demonstrates practical work across:

time-series forecasting
feature engineering
machine learning
deep learning
model comparison
horizon-wise model selection
feature stores
explainable AI
AQI intelligence
health-oriented alerting
API serving
automated pipelines
MLOps monitoring
reproducible artifacts
cloud deployment

The project therefore extends beyond a conventional model-training notebook into an end-to-end machine-learning system.

🏆 Why This Project Matters

A conventional air-quality ML project might stop at:

Dataset → Model → Accuracy

AirSense-AI extends the workflow to:

Data
 ↓
Feature Engineering
 ↓
Feature Store
 ↓
Multiple Models
 ↓
Validation & Model Selection
 ↓
72-Hour Forecast
 ↓
AQI Intelligence
 ↓
Health Alerts
 ↓
Explainability
 ↓
FastAPI
 ↓
Streamlit
 ↓
Automation & MLOps

This demonstrates experience across:

Machine Learning + Deep Learning + Data Engineering + MLOps + Backend Development + Deployment

📈 Future Improvements

Potential extensions include:

multi-sensor forecasting across Islamabad
weather-aware forecasting features
uncertainty intervals
probabilistic forecasting
Transformer-based forecasting
automated model retraining
model drift detection
real-time alert delivery
cloud-native inference
smaller/compressed model registry artifacts
additional geographic locations
mobile client integration
👨‍💻 Author
Ahmad Gul

Computer Science Undergraduate | AI / Machine Learning Engineer

Interested in building practical AI systems combining machine learning, deep learning, data engineering, explainability, and MLOps.

Connect
📧 Email: ahmadgul0310546@gmail.com
💼 LinkedIn: https://www.linkedin.com/in/ahmad-gul-8365b0307/
🐙 GitHub: https://github.com/ahmad957cs
🔗 Project Links
Resource	Link
🌫️ Live Demo	https://airsense-aigit-2w7ihp5tfvfq9jpncu8ptw.streamlit.app/

💻 GitHub Repository	https://github.com/ahmad957cs/AirSense-AI.git


📄 Academic Deliverables
 Source code
 Streamlit application
 FastAPI service
 Feature engineering pipeline
 Hopsworks integration
 GitHub Actions workflows
 Model evaluation artifacts
 Model-selection artifacts
 Verified forecast artifact
 SHAP explainability
 AQI health alerts
 MLOps dashboard
 Public GitHub repository
 Live Streamlit deployment
 Complete project report link
 Final presentation link
📜 License

This repository is primarily an academic/final-year project.

Unless a separate open-source license is added, the code should be considered an academic project rather than an unrestricted open-source software package.

🙏 Acknowledgements

AirSense-AI makes use of open-source technologies and public data infrastructure, including:

OpenAQ
Hopsworks
Python
Pandas
NumPy
Scikit-learn
TensorFlow
XGBoost
SHAP
FastAPI
Streamlit
Plotly
GitHub Actions
⭐ Support the Project

If you find AirSense-AI useful or interesting, consider starring the repository:

👉 ⭐ Star AirSense-AI on GitHub

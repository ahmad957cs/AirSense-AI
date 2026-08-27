from __future__ import annotations

import json
import os
from pathlib import Path

import hopsworks
import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

HOST = os.getenv("HOPSWORKS_HOST")
PROJECT = os.getenv("HOPSWORKS_PROJECT")
API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOST:
    raise RuntimeError("HOPSWORKS_HOST is missing.")

if not PROJECT:
    raise RuntimeError("HOPSWORKS_PROJECT is missing.")

if not API_KEY:
    raise RuntimeError("HOPSWORKS_API_KEY is missing.")


METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"

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


def mean_metrics(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)

    numeric = (
        df[["MAE", "RMSE", "R2"]]
        .mean()
        .to_dict()
    )

    return {
        key: float(value)
        for key, value in numeric.items()
    }


def connect_project():
    return hopsworks.login(
        host=HOST,
        project=PROJECT,
        api_key_value=API_KEY,
        engine="python",
    )


def register_model(
    mr,
    name: str,
    model_path: Path,
    metrics: dict[str, float],
    description: str,
):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    model = mr.sklearn.create_model(
        name=name,
        metrics=metrics,
        description=description,
    )

    registered = model.save(
        str(model_path),
        upload_configuration={
            "chunk_size": 5,
            "simultaneous_uploads": 1,
            "max_chunk_retries": 5,
        },
    )

    print(
        f"Registered {name} "
        f"version {registered.version}"
    )

    return registered


def main() -> None:
    print("=" * 70)
    print("AIRSENSE-AI — MODEL REGISTRY")
    print("=" * 70)

    project = connect_project()

    print("Connected project:", project.name)

    mr = project.get_model_registry()

    rf_metrics = mean_metrics(
        METRICS_DIR / "random_forest_validation_metrics.csv"
    )

    xgb_metrics = mean_metrics(
        METRICS_DIR / "xgboost_validation_metrics.csv"
    )

    register_model(
        mr=mr,
        name="airsense_random_forest_72h",
        model_path=RF_MODEL_PATH,
        metrics=rf_metrics,
        description=(
            "AirSense-AI 72-hour PM2.5 "
            "Random Forest forecasting model."
        ),
    )

    register_model(
        mr=mr,
        name="airsense_xgboost_72h",
        model_path=XGB_MODEL_PATH,
        metrics=xgb_metrics,
        description=(
            "AirSense-AI 72-hour PM2.5 "
            "XGBoost forecasting model."
        ),
    )

    print("\nModel registry update complete.")


if __name__ == "__main__":
    main()
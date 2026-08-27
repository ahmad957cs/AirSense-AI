from __future__ import annotations

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

XGB_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
    / "xgboost_72h.joblib"
)


def mean_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}"
        )

    df = pd.read_csv(path)

    required = ["MAE", "RMSE", "R2"]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing metric columns: {missing}"
        )

    return {
        key: float(df[key].mean())
        for key in required
    }


def connect_project():
    return hopsworks.login(
        host=HOST,
        project=PROJECT,
        api_key_value=API_KEY,
        engine="python",
    )


def main() -> None:
    print("=" * 70)
    print("AIRSENSE-AI — MODEL REGISTRY")
    print("=" * 70)

    project = connect_project()

    print("Connected project:", project.name)

    if project.name != PROJECT:
        raise RuntimeError(
            f"Unexpected project: {project.name}"
        )

    mr = project.get_model_registry()

    xgb_metrics = mean_metrics(
        METRICS_DIR
        / "xgboost_validation_metrics.csv"
    )

    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {XGB_MODEL_PATH}"
        )

    print("\nRegistering model:")
    print("airsense_xgboost_72h")
    print(
        "Artifact size:",
        f"{XGB_MODEL_PATH.stat().st_size:,}",
        "bytes",
    )

    model = mr.sklearn.create_model(
        name="airsense_xgboost_72h",
        metrics=xgb_metrics,
        description=(
            "AirSense-AI 72-hour PM2.5 "
            "XGBoost forecasting model."
        ),
    )

    registered = model.save(
        str(XGB_MODEL_PATH),
        keep_original_files=True,
        upload_configuration={
            "chunk_size": 1024,
            "simultaneous_uploads": 1,
            "max_chunk_retries": 5,
        },
    )

    print(
        f"\nRegistered airsense_xgboost_72h "
        f"version {registered.version}"
    )

    print("\nModel registry update complete.")


if __name__ == "__main__":
    main()
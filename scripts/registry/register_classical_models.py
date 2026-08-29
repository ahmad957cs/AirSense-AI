from __future__ import annotations

import os
from pathlib import Path

import hopsworks
import joblib
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# AirSense-AI — XGBoost Model Registry
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# Environment
# ============================================================

HOST = os.getenv("HOPSWORKS_HOST")
PROJECT = os.getenv("HOPSWORKS_PROJECT")
API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOST:
    raise RuntimeError("HOPSWORKS_HOST is missing.")

if not PROJECT:
    raise RuntimeError("HOPSWORKS_PROJECT is missing.")

if not API_KEY:
    raise RuntimeError("HOPSWORKS_API_KEY is missing.")


# ============================================================
# Paths
# ============================================================

METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"

XGB_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
    / "xgboost_72h.joblib"
)

XGB_REGISTRY_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost"
    / "xgboost_72h_registry.joblib"
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "airsense_xgboost_72h"

STAGING_DIR = "Resources/airsense_registry_staging"

DESCRIPTION = (
    "AirSense-AI 72-hour PM2.5 "
    "XGBoost forecasting model."
)


# ============================================================
# Metrics
# ============================================================

def mean_metrics(path: Path) -> dict[str, float]:
    """Load validation metrics and calculate column means."""

    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "MAE",
        "RMSE",
        "R2",
    ]

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
        column: float(df[column].mean())
        for column in required
    }


# ============================================================
# Hopsworks connection
# ============================================================

def connect_project():
    """Connect to the configured Hopsworks project."""

    return hopsworks.login(
        host=HOST,
        project=PROJECT,
        api_key_value=API_KEY,
        engine="python",
    )


# ============================================================
# Compress XGBoost artifact
# ============================================================

def create_registry_artifact() -> Path:
    """
    Load the original XGBoost model and create a compressed
    artifact specifically for model-registry upload.
    """

    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {XGB_MODEL_PATH}"
        )

    original_size_mb = (
        XGB_MODEL_PATH.stat().st_size
        / 1024
        / 1024
    )

    print(
        "Original XGBoost artifact:"
    )

    print(
        f"Path: {XGB_MODEL_PATH}"
    )

    print(
        f"Original size: {original_size_mb:.2f} MB"
    )

    print(
        "\nCreating compressed registry artifact..."
    )

    xgb_model = joblib.load(
        XGB_MODEL_PATH
    )

    joblib.dump(
        xgb_model,
        XGB_REGISTRY_MODEL_PATH,
        compress=3,
    )

    compressed_size_mb = (
        XGB_REGISTRY_MODEL_PATH.stat().st_size
        / 1024
        / 1024
    )

    print(
        f"Compressed size: {compressed_size_mb:.2f} MB"
    )

    return XGB_REGISTRY_MODEL_PATH


# ============================================================
# Upload to HopsFS
# ============================================================

def upload_to_hopsfs(
    project,
    model_path: Path,
) -> str:
    """
    Upload the compressed model artifact to HopsFS.

    The returned path is then passed to Model.save(), which
    supports remote HopsFS model paths.
    """

    print(
        "\nUploading registry artifact to HopsFS..."
    )

    dataset_api = (
        project.get_dataset_api()
    )

    remote_path = dataset_api.upload(
        str(model_path),
        STAGING_DIR,
        overwrite=True,
    )

    print(
        f"HopsFS staging path: {remote_path}"
    )

    return remote_path


# ============================================================
# Register model
# ============================================================

def register_model(
    project,
    mr,
    remote_model_path: str,
    metrics: dict[str, float],
):
    """Create model metadata and register the HopsFS artifact."""

    print(
        f"\nCreating model metadata: {MODEL_NAME}"
    )

    model = mr.sklearn.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        description=DESCRIPTION,
    )

    print(
        "Registering model from HopsFS..."
    )

    registered = model.save(
        remote_model_path,
        keep_original_files=True,
    )

    print(
        "\n============================================================"
    )

    print(
        "MODEL REGISTRATION SUCCESSFUL"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Version: {registered.version}"
    )

    print(
        "============================================================"
    )

    return registered


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("AIRSENSE-AI — MODEL REGISTRY")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Connect
    # --------------------------------------------------------

    print(
        "\n[1/5] Connecting to Hopsworks..."
    )

    project = connect_project()

    print(
        "Connected project:",
        project.name,
    )

    if project.name != PROJECT:
        raise RuntimeError(
            f"Unexpected project: {project.name}"
        )

    # --------------------------------------------------------
    # 2. Get Model Registry
    # --------------------------------------------------------

    print(
        "\n[2/5] Connecting to Model Registry..."
    )

    mr = project.get_model_registry()

    print(
        "Model Registry connected."
    )

    # --------------------------------------------------------
    # 3. Load metrics
    # --------------------------------------------------------

    print(
        "\n[3/5] Loading validation metrics..."
    )

    xgb_metrics = mean_metrics(
        METRICS_DIR
        / "xgboost_validation_metrics.csv"
    )

    print(
        "Metrics:",
        xgb_metrics,
    )

    # --------------------------------------------------------
    # 4. Create compressed artifact
    # --------------------------------------------------------

    print(
        "\n[4/5] Preparing registry artifact..."
    )

    registry_artifact = (
        create_registry_artifact()
    )

    # --------------------------------------------------------
    # 5. Upload + register
    # --------------------------------------------------------

    print(
        "\n[5/5] Uploading and registering..."
    )

    remote_path = upload_to_hopsfs(
        project=project,
        model_path=registry_artifact,
    )

    register_model(
        project=project,
        mr=mr,
        remote_model_path=remote_path,
        metrics=xgb_metrics,
    )

    print(
        "\nModel registry operation complete."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
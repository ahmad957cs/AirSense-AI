import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

# ============================================================
# AirSense-AI
# Hopsworks Feature Store Ingestion
# ============================================================

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY not found in .env")

DATA_PATH = "data/processed/pm25_features.csv"

PROJECT_NAME = "AirSense_AI"

FEATURE_GROUP_NAME = "pm25_hourly_features"
FEATURE_GROUP_VERSION = 3


def main():

    print("=" * 60)
    print("AIRSENSE-AI — HOPSWORKS FEATURE STORE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("\n[1/5] Loading feature dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # 2. Basic validation
    # --------------------------------------------------------

    print("\n[2/5] Validating dataset...")

    required_columns = [
        "timestamp",
        "pm25",
        "sensor_id",
        "original_observation",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if df["pm25"].isna().any():
        raise ValueError("PM2.5 contains missing values.")

    print("Validation: PASSED")

    # --------------------------------------------------------
    # 3. Connect to Hopsworks
    # --------------------------------------------------------

    print("\n[3/5] Connecting to Hopsworks...")

    project = hopsworks.login(
        host=os.getenv("HOPSWORKS_HOST"),
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        engine="python",
        cert_folder=r"D:\AirSense-AI\.hopsworks\certs",
    )

    print(f"Connected to project: {project.name}")

    fs = project.get_feature_store()

    print("Feature Store connected.")

    # --------------------------------------------------------
    # 4. Create / retrieve Feature Group
    # --------------------------------------------------------

    print("\n[4/5] Creating Feature Group...")

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=(
            "Hourly PM2.5 observations and engineered "
            "time-series features for AirSense-AI forecasting."
        ),
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="DELTA",
    )

    print(
        f"Feature Group: "
        f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}"
    )

    # --------------------------------------------------------
    # 5. Insert data
    # --------------------------------------------------------

    print("\n[5/5] Inserting feature data...")

    job, validation_report = feature_group.insert(df)

    print("\n" + "=" * 60)
    print("FEATURE STORE INGESTION COMPLETED")
    print("=" * 60)

    print(f"Rows uploaded: {len(df)}")
    print(f"Feature Group: {FEATURE_GROUP_NAME}")
    print(f"Version: {FEATURE_GROUP_VERSION}")


if __name__ == "__main__":
    main()
from pathlib import Path
import json
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# AirSense-AI
# Recreate LSTM preprocessing artifacts
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "airsense_72hour_forecasting_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessing"
)

FEATURE_SCALER_PATH = OUTPUT_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = OUTPUT_DIR / "target_scaler.pkl"
FEATURE_COLUMNS_PATH = OUTPUT_DIR / "feature_columns.json"


FEATURE_COLUMNS = [
    "coverage_percent",
    "minimum",
    "maximum",
    "average",
    "original_observation",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "pm25_lag_1h",
    "pm25_lag_3h",
    "pm25_lag_6h",
    "pm25_lag_12h",
    "pm25_lag_24h",
    "pm25_lag_48h",
    "pm25_lag_72h",
    "pm25_rolling_mean_3h",
    "pm25_rolling_std_3h",
    "pm25_rolling_min_3h",
    "pm25_rolling_max_3h",
    "pm25_rolling_mean_6h",
    "pm25_rolling_std_6h",
    "pm25_rolling_min_6h",
    "pm25_rolling_max_6h",
    "pm25_rolling_mean_12h",
    "pm25_rolling_std_12h",
    "pm25_rolling_min_12h",
    "pm25_rolling_max_12h",
    "pm25_rolling_mean_24h",
    "pm25_rolling_std_24h",
    "pm25_rolling_min_24h",
    "pm25_rolling_max_24h",
    "pm25_rolling_mean_72h",
    "pm25_rolling_std_72h",
    "pm25_rolling_min_72h",
    "pm25_rolling_max_72h",
]

TARGET_COLUMNS = [
    f"target_pm25_h{i:02d}"
    for i in range(1, 73)
]


def main() -> None:
    print("=" * 60)
    print("AIRSENSE-AI — RECREATE LSTM PREPROCESSING")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Verify required columns.
    required = (
        ["timestamp"]
        + FEATURE_COLUMNS
        + TARGET_COLUMNS
    )

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Same chronological split used for LSTM.
    n = len(df)

    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Total rows: {n}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")

    # Fit ONLY on training data.
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_scaler.fit(
        train_df[FEATURE_COLUMNS]
    )

    target_scaler.fit(
        train_df[TARGET_COLUMNS]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with FEATURE_SCALER_PATH.open("wb") as f:
        pickle.dump(feature_scaler, f)

    with TARGET_SCALER_PATH.open("wb") as f:
        pickle.dump(target_scaler, f)

    metadata = {
        "forecast_horizon_hours": 72,
        "input_window_hours": 72,
        "num_features": len(FEATURE_COLUMNS),
        "num_targets": len(TARGET_COLUMNS),
        "split": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15
        },
        "scaler": "StandardScaler",
        "fit_on": "training_data_only",
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS
    }

    with FEATURE_COLUMNS_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print("\nArtifacts created:")
    print(FEATURE_SCALER_PATH)
    print(TARGET_SCALER_PATH)
    print(FEATURE_COLUMNS_PATH)

    # Verification.
    with FEATURE_SCALER_PATH.open("rb") as f:
        loaded_feature_scaler = pickle.load(f)

    with TARGET_SCALER_PATH.open("rb") as f:
        loaded_target_scaler = pickle.load(f)

    print("\nVerification:")
    print(
        "Feature scaler type:",
        type(loaded_feature_scaler).__name__
    )
    print(
        "Target scaler type:",
        type(loaded_target_scaler).__name__
    )

    print("\n✅ LSTM preprocessing artifacts recreated.")


if __name__ == "__main__":
    main()
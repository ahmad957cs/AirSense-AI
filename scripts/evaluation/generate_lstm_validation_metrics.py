from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# AirSense-AI — Recreate LSTM Validation Metrics
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "airsense_72hour_forecasting_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lstm"
    / "airsense_lstm_72h_best.keras"
)

PREPROCESSING_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessing"
)

FEATURE_SCALER_PATH = (
    PREPROCESSING_DIR
    / "feature_scaler.pkl"
)

TARGET_SCALER_PATH = (
    PREPROCESSING_DIR
    / "target_scaler.pkl"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
)

OUTPUT_PATH = (
    METRICS_DIR
    / "lstm_validation_metrics.csv"
)


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

INPUT_WINDOW = 72


def build_sequences(
    dataframe: pd.DataFrame,
    feature_scaler,
    target_scaler,
):
    dataframe = (
        dataframe.sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    timestamps = dataframe["timestamp"]

    X_scaled = feature_scaler.transform(
        dataframe[FEATURE_COLUMNS]
    )

    Y_scaled = target_scaler.transform(
        dataframe[TARGET_COLUMNS]
    )

    X_sequences = []
    Y_sequences = []

    for i in range(INPUT_WINDOW, len(dataframe)):

        start = i - INPUT_WINDOW

        history_times = timestamps.iloc[start:i]

        diffs = history_times.diff().dropna()

        if not (
            diffs == pd.Timedelta(hours=1)
        ).all():
            continue

        X_sequences.append(
            X_scaled[start:i]
        )

        Y_sequences.append(
            Y_scaled[i]
        )

    return (
        np.asarray(X_sequences, dtype=np.float32),
        np.asarray(Y_sequences, dtype=np.float32),
    )


def main() -> None:

    print("=" * 60)
    print("AIRSENSE-AI — LSTM VALIDATION METRICS")
    print("=" * 60)

    for path in [
        DATA_PATH,
        MODEL_PATH,
        FEATURE_SCALER_PATH,
        TARGET_SCALER_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found:\n{path}"
            )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Same chronological split as training
    # --------------------------------------------------------

    n = len(df)

    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()

    # Same feature cleaning used for LSTM
    train_df = (
        train_df.dropna(subset=FEATURE_COLUMNS)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    val_df = (
        val_df.dropna(subset=FEATURE_COLUMNS)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    with FEATURE_SCALER_PATH.open("rb") as f:
        feature_scaler = pickle.load(f)

    with TARGET_SCALER_PATH.open("rb") as f:
        target_scaler = pickle.load(f)

    # --------------------------------------------------------
    # Build validation sequences
    # --------------------------------------------------------

    X_val, Y_val_scaled = build_sequences(
        val_df,
        feature_scaler,
        target_scaler,
    )

    if len(X_val) == 0:
        raise RuntimeError(
            "No valid LSTM validation sequences were created."
        )

    print("Validation sequences:", X_val.shape)
    print("Validation targets:", Y_val_scaled.shape)

    # --------------------------------------------------------
    # Load trained LSTM
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("LSTM loaded successfully.")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions_scaled = model.predict(
        X_val,
        verbose=0,
    )

    predictions = target_scaler.inverse_transform(
        predictions_scaled
    )

    actual = target_scaler.inverse_transform(
        Y_val_scaled
    )

    # --------------------------------------------------------
    # Metrics per forecast hour
    # --------------------------------------------------------

    rows = []

    for h in range(72):

        y_true = actual[:, h]
        y_pred = predictions[:, h]

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        r2 = r2_score(
            y_true,
            y_pred,
        )

        if h < 24:
            day = "Day 1"
        elif h < 48:
            day = "Day 2"
        else:
            day = "Day 3"

        rows.append(
            {
                "model": "LSTM",
                "hour_ahead": h + 1,
                "day": day,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    metrics = pd.DataFrame(rows)

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 60)
    print("LSTM VALIDATION METRICS CREATED")
    print("=" * 60)

    print(
        metrics.groupby("day")[
            ["MAE", "RMSE", "R2"]
        ].mean().to_string()
    )

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
from pathlib import Path
import json
import pickle

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# AirSense-AI
# Fair Same-Timestamp Model Evaluation
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "airsense_72hour_forecasting_dataset.csv"
)

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

LSTM_MODEL_PATH = (
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

TABULAR_IMPUTER_PATH = (
    PREPROCESSING_DIR
    / "tabular_imputer.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "model_selection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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


def make_day(hour_ahead: int) -> str:
    if hour_ahead <= 24:
        return "Day 1"
    if hour_ahead <= 48:
        return "Day 2"
    return "Day 3"


def create_lstm_sequences(
    data: pd.DataFrame,
    feature_scaler,
    target_scaler,
):
    data = (
        data.dropna(subset=FEATURE_COLUMNS)
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    timestamps = data["timestamp"]

    X_scaled = feature_scaler.transform(
        data[FEATURE_COLUMNS]
    )

    Y_scaled = target_scaler.transform(
        data[TARGET_COLUMNS]
    )

    X_sequences = []
    Y_values = []
    origin_times = []

    for i in range(INPUT_WINDOW, len(data)):

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

        Y_values.append(
            Y_scaled[i]
        )

        origin_times.append(
            timestamps.iloc[i]
        )

    return (
        np.asarray(X_sequences, dtype=np.float32),
        np.asarray(Y_values, dtype=np.float32),
        pd.DatetimeIndex(origin_times),
    )


def metric_rows(
    model_name: str,
    timestamps,
    y_true,
    predictions,
):
    rows = []

    for h in range(72):

        actual = y_true[:, h]
        predicted = predictions[:, h]

        rows.append(
            {
                "model": model_name,
                "hour_ahead": h + 1,
                "day": make_day(h + 1),
                "MAE": mean_absolute_error(
                    actual,
                    predicted,
                ),
                "RMSE": np.sqrt(
                    mean_squared_error(
                        actual,
                        predicted,
                    )
                ),
                "R2": r2_score(
                    actual,
                    predicted,
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_split(
    split_name,
    split_df,
    rf_model,
    xgb_model,
    lstm_model,
    imputer,
    feature_scaler,
    target_scaler,
):
    print("\n" + "=" * 60)
    print(f"{split_name.upper()} — COMMON TIMESTAMP EVALUATION")
    print("=" * 60)

    split_df = (
        split_df.sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    # --------------------------------------------------------
    # LSTM valid origins
    # --------------------------------------------------------

    (
        X_lstm,
        Y_lstm_scaled,
        lstm_times,
    ) = create_lstm_sequences(
        split_df,
        feature_scaler,
        target_scaler,
    )

    # Convert LSTM targets back to original units.
    Y_lstm = target_scaler.inverse_transform(
        Y_lstm_scaled
    )

    lstm_time_set = set(
        lstm_times.to_pydatetime()
    )

    # --------------------------------------------------------
    # Tree-model origins
    # --------------------------------------------------------

    tree_df = split_df[
        split_df["timestamp"].isin(lstm_times)
    ].copy()

    tree_df = tree_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # Safety check.
    if len(tree_df) != len(lstm_times):
        raise RuntimeError(
            "Tree/LSTM timestamp alignment failed."
        )

    X_tree = imputer.transform(
        tree_df[FEATURE_COLUMNS]
    )

    Y_tree = tree_df[TARGET_COLUMNS].to_numpy()

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    rf_pred = rf_model.predict(
        X_tree
    )

    xgb_pred = xgb_model.predict(
        X_tree
    )

    lstm_pred_scaled = lstm_model.predict(
        X_lstm,
        verbose=0,
    )

    lstm_pred = target_scaler.inverse_transform(
        lstm_pred_scaled
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rf_metrics = metric_rows(
        "Random Forest",
        lstm_times,
        Y_tree,
        rf_pred,
    )

    xgb_metrics = metric_rows(
        "XGBoost",
        lstm_times,
        Y_tree,
        xgb_pred,
    )

    lstm_metrics = metric_rows(
        "LSTM",
        lstm_times,
        Y_lstm,
        lstm_pred,
    )

    metrics = pd.concat(
        [
            rf_metrics,
            xgb_metrics,
            lstm_metrics,
        ],
        ignore_index=True,
    )

    metrics["split"] = split_name

    print("\nCommon timestamps:", len(lstm_times))
    print(
        "First:",
        lstm_times.min()
    )
    print(
        "Last:",
        lstm_times.max()
    )

    print("\nDaily comparison:")

    print(
        metrics.groupby(
            ["model", "day"]
        )[["MAE", "RMSE", "R2"]]
        .mean()
        .to_string()
    )

    return metrics


def main():
    print("=" * 60)
    print("AIRSENSE-AI — FAIR MODEL COMPARISON")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Same chronological split
    # --------------------------------------------------------

    n = len(df)

    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    val_df = df.iloc[
        train_end:val_end
    ].copy()

    test_df = df.iloc[
        val_end:
    ].copy()

    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    rf_model = joblib.load(
        RF_MODEL_PATH
    )

    xgb_model = joblib.load(
        XGB_MODEL_PATH
    )

    lstm_model = tf.keras.models.load_model(
        LSTM_MODEL_PATH
    )

    imputer = joblib.load(
        TABULAR_IMPUTER_PATH
    )

    with FEATURE_SCALER_PATH.open("rb") as f:
        feature_scaler = pickle.load(f)

    with TARGET_SCALER_PATH.open("rb") as f:
        target_scaler = pickle.load(f)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_metrics = evaluate_split(
        "validation",
        val_df,
        rf_model,
        xgb_model,
        lstm_model,
        imputer,
        feature_scaler,
        target_scaler,
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_metrics = evaluate_split(
        "test",
        test_df,
        rf_model,
        xgb_model,
        lstm_model,
        imputer,
        feature_scaler,
        target_scaler,
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    all_metrics = pd.concat(
        [
            validation_metrics,
            test_metrics,
        ],
        ignore_index=True,
    )

    all_metrics.to_csv(
        OUTPUT_DIR
        / "common_timestamp_model_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final selection from validation only
    # --------------------------------------------------------

    validation_only = validation_metrics.copy()

    winners = []

    for hour in range(1, 73):

        current = validation_only[
            validation_only["hour_ahead"] == hour
        ].copy()

        best_idx = current["RMSE"].idxmin()

        best = current.loc[best_idx]

        winners.append(
            {
                "hour_ahead": hour,
                "day": make_day(hour),
                "selected_model": best["model"],
                "validation_RMSE": best["RMSE"],
                "validation_MAE": best["MAE"],
                "validation_R2": best["R2"],
            }
        )

    selection = pd.DataFrame(winners)

    selection.to_csv(
        OUTPUT_DIR
        / "final_hourly_model_selection.csv",
        index=False,
    )

    model_map = {
        f"hour_{row.hour_ahead:02d}":
            row.selected_model
        for row in selection.itertuples()
    }

    with (
        OUTPUT_DIR
        / "final_model_selection.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            model_map,
            f,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("FINAL VALIDATION-BASED MODEL SELECTION")
    print("=" * 60)

    print(
        selection.to_string(index=False)
    )

    print("\nSelection counts:")

    print(
        selection["selected_model"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Evaluate selected ensemble on test
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL SELECTED STRATEGY — TEST RESULTS")
    print("=" * 60)

    # Map test metrics.
    selected_test_rows = []

    for row in selection.itertuples():

        match = test_metrics[
            (
                test_metrics["hour_ahead"]
                == row.hour_ahead
            )
            &
            (
                test_metrics["model"]
                == row.selected_model
            )
        ]

        if match.empty:
            raise RuntimeError(
                "Selected model test metric missing "
                f"for hour {row.hour_ahead}"
            )

        selected_test_rows.append(
            match.iloc[0].to_dict()
        )

    selected_test = pd.DataFrame(
        selected_test_rows
    )

    print(
        selected_test[
            ["day", "MAE", "RMSE", "R2"]
        ]
        .groupby("day")
        .mean()
        .to_string()
    )

    selected_test.to_csv(
        OUTPUT_DIR
        / "final_selected_strategy_test_metrics.csv",
        index=False,
    )

    print("\nArtifacts saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
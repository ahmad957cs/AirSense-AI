from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor


# ============================================================
# AirSense-AI
# Reproduce Classical Forecasting Models
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

IMPUTER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessing"
    / "tabular_imputer.joblib"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
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


def calculate_metrics(
    y_true: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
) -> pd.DataFrame:

    rows = []

    for h in range(72):

        actual = y_true.iloc[:, h].to_numpy()
        predicted = predictions[:, h]

        mae = mean_absolute_error(
            actual,
            predicted,
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )

        r2 = r2_score(
            actual,
            predicted,
        )

        if h < 24:
            day = "Day 1"
        elif h < 48:
            day = "Day 2"
        else:
            day = "Day 3"

        rows.append(
            {
                "model": model_name,
                "hour_ahead": h + 1,
                "day": day,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:

    print("=" * 60)
    print("AIRSENSE-AI — REPRODUCE CLASSICAL MODELS")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"],
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    required_columns = (
        ["timestamp"]
        + FEATURE_COLUMNS
        + TARGET_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Exact chronological split
    # --------------------------------------------------------

    n = len(df)

    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print("\nSplit:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    # --------------------------------------------------------
    # Prepare X/Y
    # --------------------------------------------------------

    X_train = train_df[FEATURE_COLUMNS]
    Y_train = train_df[TARGET_COLUMNS]

    X_val = val_df[FEATURE_COLUMNS]
    Y_val = val_df[TARGET_COLUMNS]

    X_test = test_df[FEATURE_COLUMNS]
    Y_test = test_df[TARGET_COLUMNS]

    # --------------------------------------------------------
    # Fit preprocessing on training data only
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_processed = imputer.fit_transform(
        X_train
    )

    X_val_processed = imputer.transform(
        X_val
    )

    X_test_processed = imputer.transform(
        X_test
    )

    IMPUTER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        imputer,
        IMPUTER_PATH,
    )

    print("\n✅ Tabular imputer saved.")

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    rf_base = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    rf_model = MultiOutputRegressor(
        rf_base,
        n_jobs=-1,
    )

    start = time.time()

    rf_model.fit(
        X_train_processed,
        Y_train,
    )

    print(
        f"Training time: {time.time() - start:.2f} seconds"
    )

    RF_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        rf_model,
        RF_MODEL_PATH,
    )

    rf_val_pred = rf_model.predict(
        X_val_processed
    )

    rf_test_pred = rf_model.predict(
        X_test_processed
    )

    rf_val_metrics = calculate_metrics(
        Y_val,
        rf_val_pred,
        "Random Forest",
    )

    rf_test_metrics = calculate_metrics(
        Y_test,
        rf_test_pred,
        "Random Forest",
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    xgb_base = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
    )

    xgb_model = MultiOutputRegressor(
        xgb_base,
        n_jobs=1,
    )

    start = time.time()

    xgb_model.fit(
        X_train_processed,
        Y_train,
    )

    print(
        f"Training time: {time.time() - start:.2f} seconds"
    )

    XGB_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        xgb_model,
        XGB_MODEL_PATH,
    )

    xgb_val_pred = xgb_model.predict(
        X_val_processed
    )

    xgb_test_pred = xgb_model.predict(
        X_test_processed
    )

    xgb_val_metrics = calculate_metrics(
        Y_val,
        xgb_val_pred,
        "XGBoost",
    )

    xgb_test_metrics = calculate_metrics(
        Y_test,
        xgb_test_pred,
        "XGBoost",
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rf_val_metrics.to_csv(
        METRICS_DIR
        / "random_forest_validation_metrics.csv",
        index=False,
    )

    rf_test_metrics.to_csv(
        METRICS_DIR
        / "random_forest_test_metrics.csv",
        index=False,
    )

    xgb_val_metrics.to_csv(
        METRICS_DIR
        / "xgboost_validation_metrics.csv",
        index=False,
    )

    xgb_test_metrics.to_csv(
        METRICS_DIR
        / "xgboost_test_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print summaries
    # --------------------------------------------------------

    validation_summary = (
        pd.concat(
            [
                rf_val_metrics,
                xgb_val_metrics,
            ],
            ignore_index=True,
        )
        .groupby(["model", "day"])[
            ["MAE", "RMSE", "R2"]
        ]
        .mean()
        .reset_index()
    )

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(
        validation_summary.to_string(
            index=False
        )
    )

    test_summary = (
        pd.concat(
            [
                rf_test_metrics,
                xgb_test_metrics,
            ],
            ignore_index=True,
        )
        .groupby(["model", "day"])[
            ["MAE", "RMSE", "R2"]
        ]
        .mean()
        .reset_index()
    )

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    print(
        test_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("ARTIFACTS")
    print("=" * 60)

    print("Random Forest:", RF_MODEL_PATH)
    print("XGBoost:", XGB_MODEL_PATH)
    print("Imputer:", IMPUTER_PATH)


if __name__ == "__main__":
    main()
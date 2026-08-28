from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# AirSense-AI — XGBoost SHAP Explainability
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
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

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pm25_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "explainability"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "xgboost_shap_importance.csv"
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


def main() -> None:

    print("=" * 70)
    print("AIRSENSE-AI — XGBOOST SHAP EXPLAINABILITY")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate artifacts
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {MODEL_PATH}"
        )

    if not IMPUTER_PATH.exists():
        raise FileNotFoundError(
            f"Imputer not found: {IMPUTER_PATH}"
        )

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_PATH}"
        )

    print("\n[1/5] Loading XGBoost model...")

    model = joblib.load(MODEL_PATH)

    print("XGBoost model loaded.")

    print("\n[2/5] Loading preprocessing artifact...")

    imputer = joblib.load(IMPUTER_PATH)

    print("Tabular imputer loaded.")

    print("\n[3/5] Loading feature data...")

    df = pd.read_csv(
        FEATURE_PATH,
        parse_dates=["timestamp"],
    )

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required feature columns:\n"
            + "\n".join(missing)
        )

    # Use a modest sample so SHAP remains practical.
    sample = df[
        FEATURE_COLUMNS
    ].tail(256).copy()

    print(
        "Rows selected for explanation:",
        len(sample),
    )

    # --------------------------------------------------------
    # Preprocess exactly as production tabular path
    # --------------------------------------------------------

    X_imputed = imputer.transform(sample)

    X_imputed = np.asarray(
        X_imputed,
        dtype=np.float32,
    )

    X = pd.DataFrame(
        X_imputed,
        columns=FEATURE_COLUMNS,
    )

    print(
        "Prepared feature matrix:",
        X.shape,
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print("\n[4/5] Calculating SHAP values...")

    # The trained model is a multi-output model.
    # Explain the first forecast horizon to keep the
    # explanation directly tied to one concrete prediction.
    if hasattr(model, "estimators_"):

        estimator = model.estimators_[0]

    else:

        estimator = model

    explainer = shap.TreeExplainer(
        estimator
    )

    shap_values = explainer.shap_values(
        X
    )

    shap_values = np.asarray(
        shap_values
    )

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mean_abs_shap": mean_abs_shap,
        }
    )

    importance = (
        importance
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\n[5/5] Saving SHAP importance...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nTop 15 features:")

    print(
        importance.head(15).to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
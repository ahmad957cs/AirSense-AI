from __future__ import annotations

import os
from pathlib import Path

import hopsworks
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# AirSense-AI
# Build supervised training data from Feature Group v3
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

FEATURE_GROUP_NAME = "pm25_hourly_features"
FEATURE_GROUP_VERSION = 3

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_X_PATH = (
    OUTPUT_DIR
    / "training_features.csv"
)

OUTPUT_Y_PATH = (
    OUTPUT_DIR
    / "training_target.csv"
)


# ------------------------------------------------------------
# Exact model feature schema
# ------------------------------------------------------------

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


def connect_feature_store():
    if not HOPSWORKS_HOST:
        raise RuntimeError(
            "HOPSWORKS_HOST is missing."
        )

    if not HOPSWORKS_PROJECT:
        raise RuntimeError(
            "HOPSWORKS_PROJECT is missing."
        )

    if not HOPSWORKS_API_KEY:
        raise RuntimeError(
            "HOPSWORKS_API_KEY is missing."
        )

    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
        engine="python",
    )

    if project.name != "AirSense_AI":
        raise RuntimeError(
            f"Unexpected Hopsworks project: {project.name}"
        )

    return project.get_feature_store()


def main() -> None:

    print("=" * 70)
    print("AIRSENSE-AI — SUPERVISED TRAINING DATASET CREATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Connect
    # --------------------------------------------------------

    print("\n[1/6] Connecting to Hopsworks...")

    fs = connect_feature_store()

    print("Feature Store connected.")

    # --------------------------------------------------------
    # 2. Get Feature Group
    # --------------------------------------------------------

    print("\n[2/6] Retrieving Feature Group...")

    feature_group = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    print(
        f"Feature Group: "
        f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}"
    )

    # --------------------------------------------------------
    # 3. Read full historical feature data
    # --------------------------------------------------------

    print("\n[3/6] Reading historical feature data...")

    df = feature_group.read(
        dataframe_type="pandas"
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    required = (
        ["timestamp", "pm25"]
        + FEATURE_COLUMNS
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Required columns are missing from "
            "Feature Group v3:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # 4. Prepare chronological source data
    # --------------------------------------------------------

    print("\n[4/6] Preparing chronological source data...")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["pm25"] = pd.to_numeric(
        df["pm25"],
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=[
                "timestamp",
                "pm25",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise RuntimeError(
            "No valid historical rows remain."
        )

    print(
        "First timestamp:",
        df["timestamp"].min(),
    )

    print(
        "Last timestamp:",
        df["timestamp"].max(),
    )

    # --------------------------------------------------------
    # 5. Create 72-hour future targets
    # --------------------------------------------------------

    print(
        "\n[5/6] Creating 72-hour supervised targets..."
    )

    # Never create targets across timestamp gaps.
    delta = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    df["_group"] = (
        delta.fillna(0)
        .ne(1)
        .cumsum()
    )

    training_parts = []

    for _, group in df.groupby(
        "_group",
        sort=False,
    ):

        group = (
            group
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # A valid row needs:
        # current features + 72 future hourly observations.
        if len(group) < 73:
            continue

        feature_part = group[
            ["timestamp"] + FEATURE_COLUMNS
        ].copy()

        target_part = pd.DataFrame(
            index=group.index
        )

        for horizon in range(1, 73):

            target_part[
                f"target_pm25_h{horizon:02d}"
            ] = (
                group["pm25"]
                .shift(-horizon)
            )

        combined = pd.concat(
            [
                feature_part,
                target_part,
            ],
            axis=1,
        )

        combined = combined.dropna(
            subset=TARGET_COLUMNS
        )

        if not combined.empty:
            training_parts.append(
                combined
            )

    if not training_parts:
        raise RuntimeError(
            "No valid 72-hour supervised training "
            "windows could be constructed."
        )

    training_df = pd.concat(
        training_parts,
        ignore_index=True,
    )

    training_df = (
        training_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        "Supervised rows:",
        len(training_df),
    )

    print(
        "Feature columns:",
        len(FEATURE_COLUMNS),
    )

    print(
        "Target columns:",
        len(TARGET_COLUMNS),
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print("\nFinal validation...")

    feature_nan_count_before = (
        training_df[FEATURE_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    target_nan_count = (
        training_df[TARGET_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Feature NaNs before cleaning:",
        feature_nan_count_before,
    )

    print(
        "Target NaNs:",
        target_nan_count,
    )

    if target_nan_count != 0:
        raise RuntimeError(
            "Training targets still contain NaN values."
        )

    # --------------------------------------------------------
    # Remove rows where forecasting input features are
    # incomplete. These occur naturally at the beginning
    # of contiguous blocks because lag/rolling features
    # require historical observations.
    # --------------------------------------------------------

    rows_before_feature_cleaning = len(training_df)

    training_df = training_df.dropna(
        subset=FEATURE_COLUMNS
    ).reset_index(drop=True)

    rows_removed_for_features = (
        rows_before_feature_cleaning
        - len(training_df)
    )

    print(
        "Rows removed due to incomplete features:",
        rows_removed_for_features,
    )

    if training_df.empty:
        raise RuntimeError(
            "No complete training rows remain after "
            "feature cleaning."
        )

    feature_nan_count_after = (
        training_df[FEATURE_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    target_nan_count_after = (
        training_df[TARGET_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Feature NaNs after cleaning:",
        feature_nan_count_after,
    )

    print(
        "Target NaNs after cleaning:",
        target_nan_count_after,
    )

    if feature_nan_count_after != 0:
        raise RuntimeError(
            "Feature NaNs remain after cleaning."
        )

    if target_nan_count_after != 0:
        raise RuntimeError(
            "Target NaNs remain after cleaning."
        )

    if len(FEATURE_COLUMNS) != 43:
        raise RuntimeError(
            "Unexpected feature schema."
        )

    if len(TARGET_COLUMNS) != 72:
        raise RuntimeError(
            "Unexpected target schema."
        )

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    print("\n[6/6] Saving training datasets...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X = training_df[
        ["timestamp"] + FEATURE_COLUMNS
    ].copy()

    y = training_df[
        TARGET_COLUMNS
    ].copy()

    X.to_csv(
        OUTPUT_X_PATH,
        index=False,
    )

    y.to_csv(
        OUTPUT_Y_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("TRAINING DATASET CREATED SUCCESSFULLY")
    print("=" * 70)

    print(
        "Features:",
        OUTPUT_X_PATH,
    )

    print(
        "Targets:",
        OUTPUT_Y_PATH,
    )

    print(
        "Rows:",
        len(X),
    )

    print(
        "Feature shape:",
        X.shape,
    )

    print(
        "Target shape:",
        y.shape,
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# AirSense-AI — PM2.5 Feature Engineering
# ============================================================

FEATURE_COLUMNS = [
    "coverage_percent",
    "minimum",
    "maximum",
    "average",
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


LAG_WINDOWS = [
    1,
    3,
    6,
    12,
    24,
    48,
    72,
]

ROLLING_WINDOWS = [
    3,
    6,
    12,
    24,
    72,
]


# ============================================================
# Validation
# ============================================================

def validate_raw_pm25_dataframe(
    df: pd.DataFrame,
) -> None:

    required = [
        "timestamp",
        "pm25",
        "coverage_percent",
        "minimum",
        "maximum",
        "average",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Raw PM2.5 dataframe is missing columns:\n"
            + "\n".join(missing)
        )

    if df.empty:
        raise ValueError(
            "Raw PM2.5 dataframe is empty."
        )

    if df["pm25"].isna().any():
        raise ValueError(
            "Raw PM2.5 contains missing values."
        )


# ============================================================
# Identify contiguous hourly blocks
# ============================================================

def _add_contiguous_group(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps found."
        )

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    delta_hours = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    # Start a new block whenever the data is not exactly
    # one hour apart.
    df["_contiguous_group"] = (
        delta_hours
        .fillna(0)
        .ne(1)
        .cumsum()
    )

    return df


# ============================================================
# Create time features
# ============================================================

def _add_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    ts = df["timestamp"]

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear

    # ISO week number
    df["week_of_year"] = (
        ts.dt.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(
        2
        * np.pi
        * df["hour"]
        / 24
    )

    df["hour_cos"] = np.cos(
        2
        * np.pi
        * df["hour"]
        / 24
    )

    # Cyclical day-of-week encoding
    df["dow_sin"] = np.sin(
        2
        * np.pi
        * df["day_of_week"]
        / 7
    )

    df["dow_cos"] = np.cos(
        2
        * np.pi
        * df["day_of_week"]
        / 7
    )

    return df


# ============================================================
# Lags + rolling statistics
# ============================================================

def _add_lag_and_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    for _, group_index in df.groupby(
        "_contiguous_group",
        sort=False,
    ).groups.items():

        group = (
            df.loc[group_index]
            .sort_values("timestamp")
        )

        pm25 = group["pm25"]

        # ----------------------------------------------------
        # LAGS
        # ----------------------------------------------------

        for lag in LAG_WINDOWS:

            column = (
                f"pm25_lag_{lag}h"
            )

            df.loc[
                group.index,
                column,
            ] = (
                pm25.shift(lag).values
            )

        # ----------------------------------------------------
        # ROLLING FEATURES
        #
        # IMPORTANT:
        # The validated dataset uses previous observations
        # only, so the current PM2.5 value is excluded.
        # ----------------------------------------------------

        previous_pm25 = pm25.shift(1)

        for window in ROLLING_WINDOWS:

            rolling = (
                previous_pm25
                .rolling(
                    window=window,
                    min_periods=window,
                )
            )

            df.loc[
                group.index,
                f"pm25_rolling_mean_{window}h",
            ] = (
                rolling.mean().values
            )

            df.loc[
                group.index,
                f"pm25_rolling_std_{window}h",
            ] = (
                rolling.std().values
            )

            df.loc[
                group.index,
                f"pm25_rolling_min_{window}h",
            ] = (
                rolling.min().values
            )

            df.loc[
                group.index,
                f"pm25_rolling_max_{window}h",
            ] = (
                rolling.max().values
            )

    return df


# ============================================================
# Main feature builder
# ============================================================

def build_pm25_features(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the production forecasting feature schema
    from raw hourly PM2.5 observations.

    The calculation is reset across timestamp gaps so that
    lag/rolling features never cross a non-hourly gap.
    """

    validate_raw_pm25_dataframe(
        raw_df
    )

    df = raw_df.copy()

    # Ensure numeric raw fields
    numeric_columns = [
        "pm25",
        "coverage_percent",
        "minimum",
        "maximum",
        "average",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "timestamp",
            "pm25",
        ]
    )

    df = _add_contiguous_group(
        df
    )

    df = _add_time_features(
        df
    )

    df = _add_lag_and_rolling_features(
        df
    )

    # This column existed in the validated dataset and is
    # used to indicate that the value originated from an
    # actual observation rather than a forecast.
    df["original_observation"] = 1

    # Ensure metadata exists when the upstream source provides
    # only the core observation fields.
    if "city" not in df.columns:
        df["city"] = "Islamabad"

    if "country" not in df.columns:
        df["country"] = "PK"

    if "sensor_id" not in df.columns:
        df["sensor_id"] = None

    if "unit" not in df.columns:
        df["unit"] = "µg/m³"

    if "timestamp_end" not in df.columns:
        df["timestamp_end"] = (
            df["timestamp"]
            + pd.Timedelta(hours=1)
        )

    metadata_columns = [
        "timestamp",
        "timestamp_end",
        "city",
        "country",
        "sensor_id",
        "pm25",
        "unit",
        "coverage_percent",
        "minimum",
        "maximum",
        "average",
        "original_observation",
    ]

    engineered_columns = [
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

    final_columns = (
        metadata_columns
        + engineered_columns
    )

    missing_final_columns = [
        column
        for column in final_columns
        if column not in df.columns
    ]

    if missing_final_columns:
        raise ValueError(
            "Feature engineering did not produce "
            "the required columns:\n"
            + "\n".join(missing_final_columns)
        )



    df = df[
        final_columns
    ]

    return (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )
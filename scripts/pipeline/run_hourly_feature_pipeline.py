
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import hopsworks
import pandas as pd
import requests
from dotenv import load_dotenv

from src.feature_engineering.features import (
    build_pm25_features,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

load_dotenv(
    PROJECT_ROOT / ".env"
)

OPENAQ_API_KEY = os.getenv(
    "OPENAQ_API_KEY"
)

OPENAQ_SENSOR_ID = os.getenv(
    "OPENAQ_SENSOR_ID",
    "13137731",
)

OPENAQ_API_URL = (
    "https://api.openaq.org/v3/"
    f"sensors/{OPENAQ_SENSOR_ID}/hours"
)

FEATURE_GROUP_NAME = (
    "pm25_hourly_features"
)

FEATURE_GROUP_VERSION = 3

FEATURE_WINDOW_HOURS = 96


# ============================================================
# Validation
# ============================================================

def validate_environment() -> None:

    if not OPENAQ_API_KEY:
        raise ValueError(
            "OPENAQ_API_KEY is missing from .env"
        )


# ============================================================
# Fetch recent OpenAQ data
# ============================================================

def fetch_recent_openaq_data(
    hours: int = FEATURE_WINDOW_HOURS,
) -> pd.DataFrame:

    validate_environment()

    end_time = datetime.now(
        timezone.utc
    )

    start_time = (
        end_time
        - timedelta(hours=hours)
    )

    headers = {
        "X-API-Key": OPENAQ_API_KEY
    }

    params = {
        "datetime_from":
            start_time.isoformat(),

        "datetime_to":
            end_time.isoformat(),

        "limit": 100,

        "page": 1,
    }

    records = []

    print(
        f"OpenAQ sensor: {OPENAQ_SENSOR_ID}"
    )

    print(
        "OpenAQ time window: "
        f"{start_time.isoformat()} → "
        f"{end_time.isoformat()}"
    )

    while True:

        response = requests.get(
            OPENAQ_API_URL,
            headers=headers,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()

        results = payload.get(
            "results",
            []
        )

        meta = payload.get(
            "meta",
            {}
        )

        print(
            f"OpenAQ page {params['page']}: "
            f"{len(results)} results"
        )

        if not results:

            print(
                "OpenAQ returned no results for "
                "the requested time window."
            )

            print(
                f"OpenAQ metadata: {meta}"
            )

            break

        page_usable_records = 0

        for item in results:

            period = item.get(
                "period",
                {}
            )

            summary = item.get(
                "summary",
                {}
            )

            timestamp = (
                period
                .get(
                    "datetimeFrom",
                    {}
                )
                .get("utc")
            )

            timestamp_end = (
                period
                .get(
                    "datetimeTo",
                    {}
                )
                .get("utc")
            )

            value = item.get(
                "value"
            )

            if (
                timestamp is None
                or value is None
            ):
                continue

            records.append(
                {
                    "timestamp":
                        timestamp,

                    "timestamp_end":
                        timestamp_end,

                    "city":
                        "Islamabad",

                    "country":
                        "PK",

                    "sensor_id":
                        OPENAQ_SENSOR_ID,

                    "pm25":
                        value,

                    "unit":
                        item.get(
                            "parameter",
                            {}
                        ).get(
                            "units"
                        ),

                    "coverage_percent":
                        item.get(
                            "coverage",
                            {}
                        ).get(
                            "percentCoverage"
                        ),

                    "minimum":
                        summary.get("min"),

                    "maximum":
                        summary.get("max"),

                    "average":
                        summary.get("avg"),
                }
            )

            page_usable_records += 1

        print(
            f"OpenAQ page {params['page']}: "
            f"{page_usable_records} usable PM2.5 records"
        )

        # ----------------------------------------------------
        # Pagination
        # ----------------------------------------------------

        if len(results) < params["limit"]:
            break

        params["page"] += 1

        # Safety guard
        if params["page"] > 100:
            print(
                "OpenAQ pagination safety limit reached."
            )
            break

    # --------------------------------------------------------
    # No usable records
    # --------------------------------------------------------

    if not records:

        print(
            "\nNo new OpenAQ PM2.5 records are available "
            f"for sensor {OPENAQ_SENSOR_ID}."
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Build DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        records
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["timestamp_end"] = pd.to_datetime(
        df["timestamp_end"],
        utc=True,
        errors="coerce",
    )

    for column in [
        "pm25",
        "coverage_percent",
        "minimum",
        "maximum",
        "average",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Clean records
    # --------------------------------------------------------

    df = (
        df.dropna(
            subset=[
                "timestamp",
                "pm25",
            ]
        )
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if df.empty:

        print(
            "\nOpenAQ returned records, but none contained "
            "a valid timestamp and PM2.5 value."
        )

        return pd.DataFrame()

    return df


# ============================================================
# Hopsworks
# ============================================================

def connect_feature_store():

    host = os.getenv(
        "HOPSWORKS_HOST",
        "eu-west.cloud.hopsworks.ai",
    )

    project_name = os.getenv(
        "HOPSWORKS_PROJECT",
        "AirSense_AI",
    )

    api_key = os.getenv(
        "HOPSWORKS_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "HOPSWORKS_API_KEY is missing from .env"
        )

    project = hopsworks.login(
        host=host,
        project=project_name,
        api_key_value=api_key,
    )

    if project.name != project_name:

        raise RuntimeError(
            f"Connected to unexpected project: "
            f"{project.name}"
        )

    return (
        project,
        project.get_feature_store(),
    )


# ============================================================
# Existing Feature Group
# ============================================================

def get_feature_group(fs):

    return fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=(
            "Hourly PM2.5 observations and engineered "
            "time-series features for AirSense-AI."
        ),
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="DELTA",
    )


# ============================================================
# Pipeline
# ============================================================

def cast_to_feature_group_schema(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match the existing Hopsworks Feature Group v3 schema
    without changing feature meanings.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="raise",
    )

    # --------------------------------------------------------
    # Existing Feature Group expects timestamp_end as STRING
    # --------------------------------------------------------

    df["timestamp_end"] = (
        pd.to_datetime(
            df["timestamp_end"],
            utc=True,
            errors="raise",
        )
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )
    )

    # --------------------------------------------------------
    # sensor_id -> DOUBLE
    # --------------------------------------------------------

    df["sensor_id"] = pd.to_numeric(
        df["sensor_id"],
        errors="raise",
    ).astype(
        "float64"
    )

    # --------------------------------------------------------
    # BIGINT columns
    # --------------------------------------------------------

    bigint_columns = [
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "original_observation",
    ]

    for column in bigint_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="raise",
            ).astype(
                "int64"
            )

    # --------------------------------------------------------
    # Floating-point columns
    # --------------------------------------------------------

    float_columns = [
        "pm25",
        "coverage_percent",
        "minimum",
        "maximum",
        "average",
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

    for column in float_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).astype(
                "float64"
            )

    return df


def run():

    print("=" * 70)

    print(
        "AIRSENSE-AI — HOURLY FEATURE PIPELINE"
    )

    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print(
        "\n[1/5] Fetching recent OpenAQ data..."
    )

    raw_df = fetch_recent_openaq_data()

    # --------------------------------------------------------
    # No new source data is not a pipeline failure.
    # Preserve existing Feature Group data and exit cleanly.
    # --------------------------------------------------------

    if raw_df.empty:

        print(
            "\nNo new source data available."
        )

        print(
            "Existing Feature Group data remains unchanged."
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "HOURLY FEATURE PIPELINE COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            "Status: No new OpenAQ observations to process."
        )

        return

    print(
        f"Raw rows: {len(raw_df)}"
    )

    print(
        "Raw range:",
        raw_df["timestamp"].min(),
        "→",
        raw_df["timestamp"].max(),
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "\n[2/5] Building forecasting features..."
    )

    feature_df = build_pm25_features(
        raw_df
    )

    print(
        f"Feature rows: {len(feature_df)}"
    )

    print(
        f"Feature columns: {len(feature_df.columns)}"
    )

    required = [
        "timestamp",
        "pm25",
    ]

    missing = [
        col
        for col in required
        if col not in feature_df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Feature engineering failed. "
            f"Missing: {missing}"
        )

    # ========================================================
    # STEP 3
    # ========================================================

    print(
        "\n[3/5] Connecting to Hopsworks..."
    )

    project, fs = (
        connect_feature_store()
    )

    print(
        "Connected:",
        project.name
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print(
        "\n[4/5] Loading Feature Group..."
    )

    feature_group = (
        get_feature_group(fs)
    )

    print(
        f"Feature Group: "
        f"{FEATURE_GROUP_NAME} "
        f"v{FEATURE_GROUP_VERSION}"
    )

    # ========================================================
    # STEP 5
    # ========================================================

    print(
        "\n[5/5] Inserting new feature rows..."
    )

    feature_view = (
        fs.get_feature_view(
            name="pm25_forecasting_view",
            version=1,
        )
    )

    latest_view = (
        feature_view.get_batch_data(
            start_time=(
                datetime.now(timezone.utc)
                - timedelta(days=14)
            ),
            end_time=datetime.now(
                timezone.utc
            ),
            event_time=True,
            dataframe_type="pandas",
        )
    )

    event_columns = [
        column
        for column in latest_view.columns
        if column.endswith("_timestamp")
    ]

    existing_timestamps = set()

    if event_columns:

        event_column = event_columns[0]

        latest_view[event_column] = (
            pd.to_datetime(
                latest_view[event_column],
                utc=True,
                errors="coerce",
            )
        )

        existing_timestamps = set(
            latest_view[event_column]
            .dropna()
            .tolist()
        )

    new_rows = feature_df[
        ~feature_df["timestamp"].isin(
            existing_timestamps
        )
    ].copy()

    if new_rows.empty:

        print(
            "\nNo new feature rows to insert."
        )

    else:

        print(
            f"New rows before schema casting: "
            f"{len(new_rows)}"
        )

        print(
            "\nCasting rows to existing "
            "Feature Group v3 schema..."
        )

        new_rows = (
            cast_to_feature_group_schema(
                new_rows
            )
        )

        print(
            "\nSelected dtypes:"
        )

        print(
            new_rows[
                [
                    "timestamp",
                    "timestamp_end",
                    "sensor_id",
                    "hour",
                    "day_of_week",
                    "day_of_month",
                    "month",
                    "day_of_year",
                ]
            ].dtypes
        )

        print(
            "\nInserting into Hopsworks..."
        )

        job, validation_report = (
            feature_group.insert(
                new_rows
            )
        )

        print(
            "Feature insertion completed."
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "HOURLY FEATURE PIPELINE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Latest source timestamp:",
        raw_df["timestamp"].max(),
    )

    print(
        "Latest feature timestamp:",
        feature_df["timestamp"].max(),
    )


if __name__ == "__main__":
    run()


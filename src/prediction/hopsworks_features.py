from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


FEATURE_VIEW_NAME = "pm25_forecasting_view"
FEATURE_VIEW_VERSION = 1

EVENT_TIME_COLUMN = (
    "airsense_ai_pm25_hourly_features_3_timestamp"
)

PROJECT_NAME = "AirSense_AI"

# Production inference should not use stale Feature Store data.
# The hourly pipeline normally refreshes this data every hour.
MAX_FEATURE_AGE_HOURS = 6


def connect_feature_store():
    """
    Connect to the AirSense-AI Hopsworks project
    and return the Feature Store.
    """

    project = hopsworks.login(
        host=os.getenv(
            "HOPSWORKS_HOST",
            "eu-west.cloud.hopsworks.ai",
        ),
        project=os.getenv(
            "HOPSWORKS_PROJECT",
            PROJECT_NAME,
        ),
        api_key_value=os.getenv(
            "HOPSWORKS_API_KEY"
        ),
        engine="python",
    )

    if project.name != PROJECT_NAME:
        raise RuntimeError(
            f"Connected to unexpected project: "
            f"{project.name}"
        )

    return project.get_feature_store()


def get_forecasting_feature_view():
    """Retrieve the existing validated Feature View."""
    fs = connect_feature_store()

    return fs.get_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
    )


def get_latest_feature_window(hours: int = 72) -> pd.DataFrame:
    """
    Read the latest forecasting features from Hopsworks.

    The forecasting system requires a continuous 72-hour
    feature window.
    """
    if hours < 72:
        raise ValueError(
            "The forecasting system requires at least "
            "a 72-hour feature window."
        )

    feature_view = get_forecasting_feature_view()

    end_time = datetime.now(timezone.utc)

    # Read extra history so we can safely select the latest
    # continuous 72-hour window even when recent data contains gaps.
    start_time = (
        end_time - timedelta(hours=hours + 24)
    )

    df = feature_view.get_batch_data(
        start_time=start_time,
        end_time=end_time,
        event_time=True,
        dataframe_type="pandas",
    )

    if df.empty:
        raise RuntimeError(
            "Hopsworks Feature View returned no feature "
            "data for the requested period."
        )

    if EVENT_TIME_COLUMN not in df.columns:
        raise RuntimeError(
            f"Expected event-time column "
            f"'{EVENT_TIME_COLUMN}' was not returned."
        )

    df = df.rename(
        columns={
            EVENT_TIME_COLUMN: "timestamp"
        }
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate data freshness
    # --------------------------------------------------------

    latest_timestamp = df["timestamp"].max()

    current_time = datetime.now(timezone.utc)

    data_age = (
        current_time - latest_timestamp
    ).total_seconds() / 3600.0

    if data_age > MAX_FEATURE_AGE_HOURS:
        raise RuntimeError(
            "Feature Store data is stale for production inference. "
            f"Latest feature timestamp: {latest_timestamp}. "
            f"Current UTC time: {current_time}. "
            f"Data age: {data_age:.1f} hours. "
            f"Maximum allowed age: {MAX_FEATURE_AGE_HOURS} hours."
        )

    # --------------------------------------------------------
    # Find the latest continuous hourly block
    # --------------------------------------------------------

    df["_is_hourly"] = (
        df["timestamp"].diff()
        == pd.Timedelta(hours=1)
    )

    # Build block IDs whenever continuity breaks.
    df["_block_id"] = (
        ~df["_is_hourly"]
    ).cumsum()

    blocks = (
        df.groupby("_block_id", sort=True)
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            rows=("timestamp", "size"),
        )
        .reset_index()
    )

    valid_blocks = blocks[
        blocks["rows"] >= hours
    ]

    if valid_blocks.empty:
        raise RuntimeError(
            f"No continuous {hours}-hour feature window "
            "is available in Hopsworks."
        )

    latest_block_id = valid_blocks.iloc[-1]["_block_id"]

    latest = df[
        df["_block_id"] == latest_block_id
    ].tail(hours).copy()

    latest = latest.drop(
        columns=["_is_hourly", "_block_id"]
    )

    if len(latest) != hours:
        raise RuntimeError(
            f"Expected {hours} hourly rows, "
            f"received {len(latest)}."
        )

    # Final safety check.
    differences = (
        latest["timestamp"].diff().dropna()
    )

    if not (
        differences == pd.Timedelta(hours=1)
    ).all():
        raise RuntimeError(
            "Latest Hopsworks feature window is not "
            "hourly continuous."
        )

    return latest.reset_index(drop=True)
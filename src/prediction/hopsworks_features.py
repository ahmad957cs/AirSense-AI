from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import hopsworks
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FEATURE_VIEW_NAME = "pm25_forecasting_view"
FEATURE_VIEW_VERSION = 1

EVENT_TIME_COLUMN = (
    "airsense_ai_pm25_hourly_features_3_timestamp"
)

PROJECT_NAME = "AirSense_AI"


# ============================================================
# HOPSWORKS CONNECTION
# ============================================================

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
        api_key_value=os.getenv(
            "HOPSWORKS_API_KEY"
        ),
    )

    if project.name != PROJECT_NAME:
        raise RuntimeError(
            f"Connected to unexpected project: "
            f"{project.name}"
        )

    return project.get_feature_store()


# ============================================================
# FEATURE VIEW
# ============================================================

def get_forecasting_feature_view():
    """
    Retrieve the existing validated Feature View.
    Does not create or modify anything.
    """

    fs = connect_feature_store()

    feature_view = fs.get_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
    )

    return feature_view


# ============================================================
# LATEST 72 HOURS
# ============================================================

def get_latest_feature_window(
    hours: int = 72,
) -> pd.DataFrame:

    if hours < 72:
        raise ValueError(
            "The forecasting system requires "
            "at least a 72-hour feature window."
        )

    feature_view = (
        get_forecasting_feature_view()
    )

    end_time = datetime.now(
        timezone.utc
    )

    start_time = (
        end_time
        - timedelta(hours=hours + 24)
    )

    df = feature_view.get_batch_data(
        start_time=start_time,
        end_time=end_time,
        event_time=True,
        dataframe_type="pandas",
    )

    if df.empty:
        raise RuntimeError(
            "Hopsworks Feature View returned "
            "no feature data for the requested period."
        )

    # --------------------------------------------------------
    # Normalize event timestamp
    # --------------------------------------------------------

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
    )

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate hourly continuity
    # --------------------------------------------------------

    latest = df.tail(hours)

    if len(latest) < hours:
        raise RuntimeError(
            f"Only {len(latest)} hourly rows "
            f"available; {hours} required."
        )

    differences = (
        latest["timestamp"]
        .diff()
        .dropna()
    )

    if not (
        differences
        == pd.Timedelta(hours=1)
    ).all():

        raise RuntimeError(
            "Latest Hopsworks feature window "
            "is not hourly continuous."
        )

    return latest.reset_index(
        drop=True
    )
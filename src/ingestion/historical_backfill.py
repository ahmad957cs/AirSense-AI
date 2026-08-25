import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.logger import logger


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

# Islamabad OpenAQ PM2.5 sensor confirmed during our testing
OPENAQ_SENSOR_ID = os.getenv("OPENAQ_SENSOR_ID", "13137731")

# Historical period
START_DATE = os.getenv(
    "HISTORICAL_START_DATE",
    "2025-01-01T00:00:00Z"
)

END_DATE = os.getenv(
    "HISTORICAL_END_DATE",
    "2026-08-10T00:00:00Z"
)

# OpenAQ API
API_URL = (
    f"https://api.openaq.org/v3/"
    f"sensors/{OPENAQ_SENSOR_ID}/hours"
)

# Number of records requested per page
PAGE_LIMIT = 100

# Small delay between requests
REQUEST_DELAY = 0.2

# Output directory
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "historical_pm25.csv"
)


# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    """Validate required configuration."""

    if not OPENAQ_API_KEY:
        raise ValueError(
            "OPENAQ_API_KEY is missing from .env"
        )

    logger.info(
        f"OpenAQ sensor ID: {OPENAQ_SENSOR_ID}"
    )

    logger.info(
        f"Historical period: {START_DATE} → {END_DATE}"
    )


# ============================================================
# FETCH ONE PAGE
# ============================================================

def fetch_page(page: int):
    """
    Fetch one page of hourly PM2.5 data from OpenAQ.
    """

    headers = {
        "X-API-Key": OPENAQ_API_KEY
    }

    params = {
        "datetime_from": START_DATE,
        "datetime_to": END_DATE,
        "limit": PAGE_LIMIT,
        "page": page,
    }

    response = requests.get(
        API_URL,
        headers=headers,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    return data


# ============================================================
# EXTRACT RECORDS
# ============================================================

def extract_records(data):
    """
    Convert OpenAQ API response into simple records.
    """

    records = []

    for item in data.get("results", []):

        period = item.get("period", {})
        summary = item.get("summary", {})

        datetime_from = (
            period
            .get("datetimeFrom", {})
            .get("utc")
        )

        datetime_to = (
            period
            .get("datetimeTo", {})
            .get("utc")
        )

        value = item.get("value")

        if datetime_from is None or value is None:
            continue

        records.append(
            {
                "timestamp": datetime_from,
                "timestamp_end": datetime_to,
                "pm25": value,
                "unit": item.get(
                    "parameter", {}
                ).get("units"),
                "coverage_percent": (
                    item.get("coverage", {})
                    .get("percentCoverage")
                ),
                "minimum": summary.get("min"),
                "maximum": summary.get("max"),
                "average": summary.get("avg"),
            }
        )

    return records


# ============================================================
# HISTORICAL BACKFILL
# ============================================================

def run_backfill():

    validate_config()

    logger.info(
        "Starting OpenAQ historical backfill..."
    )

    logger.info(
        f"Sensor: {OPENAQ_SENSOR_ID}"
    )

    logger.info(
        f"Period: {START_DATE} → {END_DATE}"
    )

    all_records = []

    page = 1

    while True:

        logger.info(
            f"Fetching OpenAQ page {page}..."
        )

        try:

            data = fetch_page(page)

        except requests.RequestException as exc:

            logger.error(
                f"OpenAQ request failed on page "
                f"{page}: {exc}"
            )

            raise

        records = extract_records(data)

        if not records:

            logger.info(
                f"No more records found at page {page}."
            )

            break

        all_records.extend(records)

        logger.info(
            f"Page {page}: "
            f"{len(records)} records fetched | "
            f"Total: {len(all_records)}"
        )

        # Do NOT assume that fewer than PAGE_LIMIT records
        # means this is the final page.
        #
        # OpenAQ may return fewer records than requested
        # while additional pages still exist.

        meta = data.get("meta", {})

        found = meta.get("found")

        logger.info(
            f"OpenAQ reported total: {found}"
        )

        page += 1

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    if not all_records:
        raise RuntimeError(
            "Historical backfill returned zero records."
        )

    df = pd.DataFrame(all_records)

    # ========================================================
    # CLEAN TIMESTAMPS
    # ========================================================

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

    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

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

    # ========================================================
    # REMOVE INVALID DATA
    # ========================================================

    before = len(df)

    df = df.dropna(
        subset=[
            "timestamp",
            "pm25",
        ]
    )

    removed = before - len(df)

    if removed > 0:

        logger.warning(
            f"Removed {removed} invalid records."
        )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    before = len(df)

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    duplicates_removed = before - len(df)

    if duplicates_removed > 0:

        logger.info(
            f"Removed {duplicates_removed} "
            f"duplicate timestamps."
        )

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # ========================================================
    # ADD DATASET METADATA
    # ========================================================

    df["sensor_id"] = OPENAQ_SENSOR_ID
    df["city"] = "Islamabad"
    df["country"] = "PK"

    # Reorder columns
    df = df[
        [
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
        ]
    ]

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    logger.info(
        "Historical backfill completed successfully."
    )

    logger.info(
        f"Total records: {len(df)}"
    )

    logger.info(
        f"Date range: "
        f"{df['timestamp'].min()} → "
        f"{df['timestamp'].max()}"
    )

    logger.info(
        f"Output: {OUTPUT_FILE}"
    )

    print("\n========== HISTORICAL BACKFILL ==========")

    print(
        f"Records          : {len(df)}"
    )

    print(
        f"Columns          : {len(df.columns)}"
    )

    print(
        f"Start            : {df['timestamp'].min()}"
    )

    print(
        f"End              : {df['timestamp'].max()}"
    )

    print(
        f"PM2.5 Mean       : "
        f"{df['pm25'].mean():.2f}"
    )

    print(
        f"PM2.5 Min        : "
        f"{df['pm25'].min():.2f}"
    )

    print(
        f"PM2.5 Max        : "
        f"{df['pm25'].max():.2f}"
    )

    print(
        f"\nSaved to         : {OUTPUT_FILE}"
    )

    print("\n========== PREVIEW ==========")

    print(
        df.head(10).to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_backfill()
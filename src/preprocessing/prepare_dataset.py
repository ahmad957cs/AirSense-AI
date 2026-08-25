import logging
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/historical_pm25.csv")

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "pm25_features.csv"

# We will NOT interpolate long gaps.
# Short gaps up to this many consecutive hours can be interpolated.
MAX_INTERPOLATION_GAP = 6


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# GAP ANALYSIS
# ============================================================

def analyze_gaps(df):

    timestamps = df["timestamp"]

    gaps = timestamps.diff()

    one_hour = pd.Timedelta(hours=1)

    gap_rows = pd.DataFrame({
        "timestamp": timestamps,
        "gap": gaps
    })

    missing_hours = int(
        ((gaps / one_hour) - 1)
        .clip(lower=0)
        .sum()
    )

    large_gaps = gap_rows[
        gap_rows["gap"] > one_hour
    ].copy()

    logger.info("========== GAP ANALYSIS ==========")
    logger.info("Original records: %d", len(df))
    logger.info(
        "Expected missing hourly slots: %d",
        missing_hours
    )

    if not large_gaps.empty:

        logger.info(
            "Number of gaps > 1 hour: %d",
            len(large_gaps)
        )

        logger.info(
            "Largest gap: %s",
            large_gaps["gap"].max()
        )

    return missing_hours


# ============================================================
# CREATE HOURLY TIMELINE
# ============================================================

def create_hourly_timeline(df):

    df = df.set_index("timestamp")

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="1h",
        tz="UTC"
    )

    df = df.reindex(full_index)

    df.index.name = "timestamp"

    return df


# ============================================================
# GAP HANDLING
# ============================================================

def handle_gaps(df):

    logger.info("========== GAP HANDLING ==========")

    original_missing = df["pm25"].isna().sum()

    logger.info(
        "Missing PM2.5 after timeline expansion: %d",
        original_missing
    )

    # Track where observations originally existed
    df["original_observation"] = df["pm25"].notna().astype(int)

    # Identify consecutive missing runs
    missing = df["pm25"].isna()

    groups = missing.ne(
        missing.shift()
    ).cumsum()

    gap_lengths = (
        missing
        .groupby(groups)
        .transform("sum")
    )

    # Only interpolate short gaps.
    short_gap_mask = (
        missing
        & (gap_lengths <= MAX_INTERPOLATION_GAP)
    )

    before = df["pm25"].isna().sum()

    # Time interpolation only across short gaps.
    interpolated = (
        df["pm25"]
        .interpolate(
            method="time",
            limit=MAX_INTERPOLATION_GAP,
            limit_direction="both"
        )
    )

    df.loc[short_gap_mask, "pm25"] = interpolated.loc[
        short_gap_mask
    ]

    after = df["pm25"].isna().sum()

    filled = before - after

    logger.info(
        "Short-gap values filled: %d",
        filled
    )

    logger.info(
        "Long-gap values remaining missing: %d",
        after
    )

    # IMPORTANT:
    # Long gaps are deliberately NOT filled.
    # We don't fabricate 100+ days of PM2.5 values.

    return df


# ============================================================
# TIME FEATURES
# ============================================================

def create_time_features(df):

    logger.info("========== TIME FEATURES ==========")

    timestamp = df.index

    df["hour"] = timestamp.hour
    df["day_of_week"] = timestamp.dayofweek
    df["day_of_month"] = timestamp.day
    df["month"] = timestamp.month
    df["day_of_year"] = timestamp.dayofyear
    df["week_of_year"] = timestamp.isocalendar().week.astype(int)
    df["is_weekend"] = (
        timestamp.dayofweek >= 5
    ).astype(int)

    # Cyclic hour encoding
    import numpy as np

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Cyclic day-of-week encoding
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


# ============================================================
# LAG FEATURES
# ============================================================

def create_lag_features(df):

    logger.info("========== LAG FEATURES ==========")

    for lag in [1, 3, 6, 12, 24, 48, 72]:

        df[f"pm25_lag_{lag}h"] = (
            df["pm25"].shift(lag)
        )

    return df


# ============================================================
# ROLLING FEATURES
# ============================================================

def create_rolling_features(df):

    logger.info("========== ROLLING FEATURES ==========")

    # Shift first so current PM2.5 is NEVER included.
    previous = df["pm25"].shift(1)

    for window in [3, 6, 12, 24, 72]:

        df[f"pm25_rolling_mean_{window}h"] = (
            previous
            .rolling(window=window)
            .mean()
        )

        df[f"pm25_rolling_std_{window}h"] = (
            previous
            .rolling(window=window)
            .std()
        )

        df[f"pm25_rolling_min_{window}h"] = (
            previous
            .rolling(window=window)
            .min()
        )

        df[f"pm25_rolling_max_{window}h"] = (
            previous
            .rolling(window=window)
            .max()
        )

    return df


# ============================================================
# FINAL CLEANUP
# ============================================================

def finalize_dataset(df):

    logger.info("========== FINALIZING DATASET ==========")

    df = df.reset_index()

    # Keep only rows where the target exists.
    # Long temporal gaps remain represented by missing
    # observations and will not be used as training targets.
    df = df.dropna(
        subset=["pm25"]
    ).reset_index(drop=True)

    logger.info(
        "Final rows: %d",
        len(df)
    )

    logger.info(
        "Final columns: %d",
        len(df.columns)
    )

    return df


# ============================================================
# SAVE
# ============================================================

def save_dataset(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    logger.info(
        "Saved processed dataset: %s",
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("==========================================")
    logger.info("       AIRSENSE-AI DATA PREPROCESSING")
    logger.info("==========================================")

    df = load_data()

    logger.info(
        "Loaded %d original records",
        len(df)
    )

    analyze_gaps(df)

    df = create_hourly_timeline(df)

    df = handle_gaps(df)

    df = create_time_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = finalize_dataset(df)

    save_dataset(df)

    logger.info("==========================================")
    logger.info("PREPROCESSING COMPLETED")
    logger.info("==========================================")

    print()
    print("========== PREPROCESSING SUMMARY ==========")
    print("Output:", OUTPUT_FILE)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("PM2.5 missing:", df["pm25"].isna().sum())
    print("===========================================")


if __name__ == "__main__":
    main()
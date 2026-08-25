from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# AirSense-AI — Formal EDA Report Generator
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pm25_features.csv"
)

EDA_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "eda"
)

PLOTS_DIR = EDA_DIR / "plots"

EDA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / filename,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()


def write_json(filename: str, data: dict) -> None:
    with (EDA_DIR / filename).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            default=str,
        )


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("AIRSENSE-AI — FORMAL EDA")
print("=" * 70)

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

print("\nDataset:")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print(
    "First timestamp:",
    df["timestamp"].min()
)

print(
    "Last timestamp:",
    df["timestamp"].max()
)


# ============================================================
# Basic information
# ============================================================

basic_info = {
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "first_timestamp": df["timestamp"].min(),
    "last_timestamp": df["timestamp"].max(),
    "duplicate_rows": int(df.duplicated().sum()),
    "duplicate_timestamps": int(
        df["timestamp"].duplicated().sum()
    ),
    "total_missing_cells": int(
        df.isna().sum().sum()
    ),
}

write_json(
    "basic_dataset_info.json",
    basic_info,
)


# ============================================================
# Missing values
# ============================================================

missing = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
)

missing_nonzero = missing[
    missing > 0
]

missing_df = (
    missing_nonzero
    .rename("missing_count")
    .reset_index()
    .rename(columns={"index": "column"})
)

missing_df["missing_percent"] = (
    missing_df["missing_count"]
    / len(df)
    * 100
)

missing_df.to_csv(
    EDA_DIR / "missing_values.csv",
    index=False,
)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

if missing_nonzero.empty:
    print("No missing values.")
else:
    print(
        missing_df.to_string(index=False)
    )


# Missing-value chart
if not missing_df.empty:

    plt.figure(figsize=(12, 7))

    top_missing = missing_df.head(20)

    plt.bar(
        top_missing["column"],
        top_missing["missing_count"],
    )

    plt.xticks(
        rotation=70,
        ha="right",
    )

    plt.xlabel("Feature")
    plt.ylabel("Missing cells")
    plt.title("Top Features by Missing Values")

    save_plot(
        "missing_values.png"
    )


# ============================================================
# PM2.5 statistics
# ============================================================

if "pm25" in df.columns:

    pm25_stats = (
        df["pm25"]
        .describe()
        .to_dict()
    )

    write_json(
        "pm25_statistics.json",
        pm25_stats,
    )

    print("\n" + "=" * 70)
    print("PM2.5 STATISTICS")
    print("=" * 70)

    print(
        df["pm25"].describe()
    )


# ============================================================
# PM2.5 time-series
# ============================================================

if "pm25" in df.columns:

    plt.figure(figsize=(16, 6))

    plt.plot(
        df["timestamp"],
        df["pm25"],
        linewidth=0.8,
    )

    plt.xlabel("Timestamp")
    plt.ylabel("PM2.5")
    plt.title("PM2.5 Over Time")

    save_plot(
        "pm25_time_series.png"
    )


# ============================================================
# Rolling statistics
# ============================================================

if "pm25" in df.columns:

    rolling = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "pm25": df["pm25"],
            "rolling_mean_24h": (
                df["pm25"]
                .rolling(24)
                .mean()
            ),
            "rolling_std_24h": (
                df["pm25"]
                .rolling(24)
                .std()
            ),
        }
    )

    rolling.to_csv(
        EDA_DIR / "rolling_statistics.csv",
        index=False,
    )

    plt.figure(figsize=(16, 6))

    plt.plot(
        rolling["timestamp"],
        rolling["pm25"],
        label="PM2.5",
        linewidth=0.8,
    )

    plt.plot(
        rolling["timestamp"],
        rolling["rolling_mean_24h"],
        label="24h rolling mean",
        linewidth=1.5,
    )

    plt.xlabel("Timestamp")
    plt.ylabel("PM2.5")
    plt.title("PM2.5 and 24-Hour Rolling Mean")
    plt.legend()

    save_plot(
        "pm25_rolling_24h.png"
    )


# ============================================================
# Hourly pattern
# ============================================================

if "pm25" in df.columns:

    hourly = (
        df.groupby("hour")["pm25"]
        .agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max",
                "count",
            ]
        )
        .reset_index()
    )

    hourly.to_csv(
        EDA_DIR / "hourly_pm25.csv",
        index=False,
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        hourly["hour"],
        hourly["mean"],
        marker="o",
    )

    plt.xlabel("Hour of day")
    plt.ylabel("Mean PM2.5")
    plt.title("Average PM2.5 by Hour")

    plt.xticks(
        range(0, 24)
    )

    save_plot(
        "hourly_pm25_pattern.png"
    )


# ============================================================
# Day-of-week pattern
# ============================================================

if "pm25" in df.columns:

    if "day_of_week" in df.columns:

        dow = (
            df.groupby("day_of_week")["pm25"]
            .agg(
                [
                    "mean",
                    "median",
                    "std",
                    "min",
                    "max",
                    "count",
                ]
            )
            .reset_index()
        )

        dow.to_csv(
            EDA_DIR / "day_of_week_pm25.csv",
            index=False,
        )

        plt.figure(figsize=(11, 6))

        plt.bar(
            dow["day_of_week"].astype(str),
            dow["mean"],
        )

        plt.xlabel("Day of week")
        plt.ylabel("Mean PM2.5")
        plt.title(
            "Average PM2.5 by Day of Week"
        )

        save_plot(
            "day_of_week_pm25.png"
        )


# ============================================================
# Monthly pattern
# ============================================================

if "pm25" in df.columns:

    monthly = (
        df.groupby("month")["pm25"]
        .agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max",
                "count",
            ]
        )
        .reset_index()
    )

    monthly.to_csv(
        EDA_DIR / "monthly_pm25.csv",
        index=False,
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        monthly["month"],
        monthly["mean"],
        marker="o",
    )

    plt.xlabel("Month")
    plt.ylabel("Mean PM2.5")
    plt.title("Average PM2.5 by Month")

    plt.xticks(
        sorted(
            monthly["month"].dropna().unique()
        )
    )

    save_plot(
        "monthly_pm25_pattern.png"
    )


# ============================================================
# Weekend vs weekday
# ============================================================

if (
    "pm25" in df.columns
    and "is_weekend" in df.columns
):

    weekend = (
        df.groupby("is_weekend")["pm25"]
        .agg(
            [
                "mean",
                "median",
                "std",
                "count",
            ]
        )
        .reset_index()
    )

    weekend.to_csv(
        EDA_DIR / "weekend_pm25.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        weekend["is_weekend"].astype(str),
        weekend["mean"],
    )

    plt.xlabel("Weekend flag")
    plt.ylabel("Mean PM2.5")
    plt.title(
        "PM2.5: Weekday vs Weekend"
    )

    save_plot(
        "weekend_pm25.png"
    )


# ============================================================
# Distribution
# ============================================================

if "pm25" in df.columns:

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["pm25"].dropna(),
        bins=50,
    )

    plt.xlabel("PM2.5")
    plt.ylabel("Frequency")
    plt.title("PM2.5 Distribution")

    save_plot(
        "pm25_distribution.png"
    )


# ============================================================
# Boxplot / outliers
# ============================================================

if "pm25" in df.columns:

    plt.figure(figsize=(7, 6))

    plt.boxplot(
        df["pm25"].dropna(),
        vert=True,
    )

    plt.ylabel("PM2.5")
    plt.title("PM2.5 Boxplot")

    save_plot(
        "pm25_boxplot.png"
    )


# ============================================================
# Correlation analysis
# ============================================================

numeric_df = df.select_dtypes(
    include=np.number
)

correlation = (
    numeric_df.corr()
)

correlation.to_csv(
    EDA_DIR / "correlation_matrix.csv"
)

# Focus on PM2.5 correlations
if "pm25" in correlation.columns:

    pm25_corr = (
        correlation["pm25"]
        .drop("pm25")
        .sort_values(
            key=lambda s: s.abs(),
            ascending=False,
        )
        .to_frame("correlation")
    )

    pm25_corr.to_csv(
        EDA_DIR / "pm25_correlations.csv"
    )

    top_corr = pm25_corr.head(20)

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_corr.index[::-1],
        top_corr["correlation"][::-1],
    )

    plt.xlabel("Correlation with PM2.5")
    plt.ylabel("Feature")
    plt.title(
        "Top Feature Correlations with PM2.5"
    )

    save_plot(
        "pm25_feature_correlations.png"
    )


# ============================================================
# Timestamp continuity / gaps
# ============================================================

timestamp_diff = (
    df["timestamp"]
    .sort_values()
    .diff()
    .dropna()
)

gap_summary = (
    timestamp_diff
    .value_counts()
    .sort_index()
)

gap_df = (
    gap_summary
    .rename("count")
    .reset_index()
    .rename(columns={"index": "time_difference"})
)

gap_df.to_csv(
    EDA_DIR / "timestamp_gap_analysis.csv",
    index=False,
)

one_hour_count = int(
    (
        timestamp_diff
        == pd.Timedelta(hours=1)
    ).sum()
)

non_hourly_count = int(
    len(timestamp_diff)
    - one_hour_count
)

gap_info = {
    "total_intervals": int(len(timestamp_diff)),
    "one_hour_intervals": one_hour_count,
    "non_hourly_intervals": non_hourly_count,
    "largest_gap": (
        str(timestamp_diff.max())
        if not timestamp_diff.empty
        else None
    ),
}

write_json(
    "timestamp_gap_summary.json",
    gap_info,
)

print("\n" + "=" * 70)
print("TIMESTAMP CONTINUITY")
print("=" * 70)

print(
    "1-hour intervals:",
    one_hour_count
)

print(
    "Non-hourly intervals:",
    non_hourly_count
)

print(
    "Largest gap:",
    gap_info["largest_gap"]
)


# ============================================================
# Dataset-level summary
# ============================================================

summary = {
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "missing_cells": int(
        df.isna().sum().sum()
    ),
    "duplicate_rows": int(
        df.duplicated().sum()
    ),
    "duplicate_timestamps": int(
        df["timestamp"].duplicated().sum()
    ),
    "one_hour_intervals": one_hour_count,
    "non_hourly_intervals": non_hourly_count,
    "numeric_columns": int(
        len(numeric_df.columns)
    ),
    "categorical_columns": int(
        len(
            df.select_dtypes(
                exclude=np.number
            ).columns
        )
    ),
}

write_json(
    "eda_summary.json",
    summary,
)


# ============================================================
# Completion
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print("Artifacts:", EDA_DIR)
print("Plots:", PLOTS_DIR)

print("\nGenerated:")
for path in sorted(EDA_DIR.iterdir()):
    if path.is_file():
        print(" -", path.name)

print("\nPlots:")
for path in sorted(PLOTS_DIR.iterdir()):
    if path.is_file():
        print(" -", path.name)